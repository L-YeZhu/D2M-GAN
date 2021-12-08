import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import argparse
from pathlib import Path
import jukebox.utils.dist_adapter as dist

from jukebox.hparams import Hyperparams
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav, load_audio
from jukebox.make_models import make_vae_model
from jukebox.utils.sample_utils import split_batch, get_starts
from jukebox.utils.dist_utils import print_once
import fire
import librosa
import soundfile as sf 

from d2m.dataset import VAMDataset
from d2m.d2m_modules import vqEncoder_top, Discriminator, motion_encoder, Audio2Mel
from d2m.utils import save_sample



def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    args = parser.parse_args()
    return args




# load entire audio files
def load_entire(audio_files, hps):
    xs = []
    #for audio_file in audio_files:
    a, sr = librosa.load(audio_files, sr=22050, offset=0.0, mono=True)
    #print("original a", np.shape(a))
    #sf.write('original.wav', a, sr)
    #a = np.asarray(a)
    #a = a.astype(float)
    if len(a.shape) == 1:
        a = a.reshape((1,-1))
    a = a.T # ct -> tc
    xs.append(a)
    #print(xs[0].dtype)
    x = t.stack([t.from_numpy(x) for x in xs])
    x = x.to('cuda', non_blocking=True)
    return x




# Load codes from previous sampling run
def load_codes(codes_file, duration, priors, hps):
    data = t.load(codes_file, map_location='cpu')
    zs = [z.cuda() for z in data['zs']]
    assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
    del data
    if duration is not None:
        # Cut off codes to match duration
        top_raw_to_tokens = priors[-1].raw_to_tokens
        assert duration % top_raw_to_tokens == 0, f"Cut-off duration {duration} not an exact multiple of top_raw_to_tokens"
        assert duration//top_raw_to_tokens <= zs[-1].shape[1], f"Cut-off tokens {duration//priors[-1].raw_to_tokens} longer than tokens {zs[-1].shape[1]} in saved codes"
        zs = [z[:,:duration//prior.raw_to_tokens] for z, prior in zip(zs, priors)]
    return zs

# Generate and save samples, alignment, and webpage for visualization.
def train(model, device, hps, sample_hps):
    ##from jukebox.lyricdict import poems, gpt_2_lyrics
    root = '/home/zhuye/musicgen/logs'
    batch_size = 16
    #args = parse_args()
    writer = SummaryWriter(str(root))

    #### create the model ######
    num_D = 3
    ndf = 32
    n_layers_D = 4
    downsamp_factor = 4

    vqvae= make_vae_model(model, device, hps).cuda()
    encoder = vqEncoder_top().cuda()
    mencoder = motion_encoder().cuda()
    netD = Discriminator(num_D, ndf, n_layers_D, downsamp_factor).cuda()
    fft = Audio2Mel(n_mel_channels=128).cuda()
    
    print(mencoder)
    print(encoder)
    print(netD)



    
    #### create optimizer #####
    t_param = list(mencoder.parameters()) + list(encoder.parameters())
    optG = t.optim.Adam(t_param, lr=1e-4, betas=(0.5, 0.9))
    optD = t.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    #vqvae.load_state_dict(t.load("/data/zhuye/music_results/d2m_models/logs_top/top_vqvae1.pt"))
    vqvae.eval()

    #### continue training ####
    #load_root = '/home/zhuye/musicgen/logs'
    #mencoder.load_state_dict(t.load("/home/zhuye/musicgen/logs_top/mencoder.pt"))
    #encoder.load_state_dict(t.load("/home/zhuye/musicgen/logs_top/netG.pt"))
    #optG.load_state_dict(t.load("/home/zhuye/musicgen/logs_1017/optG.pt"))
    #netD.load_state_dict(t.load("/home/zhuye/musicgen/logs_1017/netD.pt"))
    #optD.load_state_dict(t.load("/home/zhuye/musicgen/logs_1017/optD.pt"))
    print("Now continue training...")



    #### creat data loader ####
    root = '/home/zhuye/musicgen'
    va_train_set = VAMDataset( audio_files = '/home/zhuye/musicgen/aist_audio_train_segment.txt', video_files = '/home/zhuye/musicgen/aist_video_train_segment.txt', genre_label = '/home/zhuye/musicgen/train_genre.npy', motion_files = '/home/zhuye/musicgen/aist_motion_train_segment.txt')
    va_train_loader = DataLoader(va_train_set, batch_size = batch_size, num_workers=4, shuffle=True)
    va_test_set = VAMDataset( audio_files = '/home/zhuye/musicgen/aist_audio_test_segment.txt', video_files = '/home/zhuye/musicgen/aist_video_test_segment.txt', genre_label = '/home/zhuye/musicgen/test_genre.npy', motion_files = '/home/zhuye/musicgen/aist_motion_test_segment.txt', augment=False)
    va_test_loader = DataLoader(va_test_set, batch_size = 1)
    print("finish data loader", len(va_train_loader), len(va_test_loader)) 
    

    #### dumping original audio ####
    test_video = []
    test_audio = []
    test_genre = []
    test_motion = []
    for i, (a_t, v_t, m_t, genre) in enumerate(va_test_loader):
        a_t = a_t.float().cuda()
        test_video.append(v_t.float().cuda())
        test_audio.append(a_t)
        test_genre.append(genre.float().cuda())
        test_motion.append(m_t.float().cuda())
        #print("label check", genre)
        gt_xs, zs_code = vqvae._encode(a_t.transpose(1,2))
        zs_middle = []
        zs_middle.append(zs_code[2])
        quantised_xs, out = vqvae._decode(zs_middle, start_level=2, end_level=3)
        audio = a_t.squeeze()
        out = out.squeeze()#.detach().cpu().numpy()
        gt_code_error = F.l1_loss(gt_xs[2], quantised_xs[0])
        print("outputsize", np.shape(out), np.shape(audio[0:44032]))
        audio_error = F.l1_loss(audio[0:44032], out)
        print("check output from vqvae", gt_xs[2].size(), quantised_xs[0].size(), gt_code_error, audio_error)
        sf.write("/home/zhuye/musicgen/samples/original_%d.wav" % (i+1), audio.detach().cpu().numpy(), 22050)
        sf.write("/home/zhuye/musicgen/samples/vqvae_%d.wav" % (i+1), out.detach().cpu().numpy(), 22050)

        if i > 8:
            break
    print("finish dumping samples", len(test_audio), len(test_video))
    #exit()

    #### start training ###
    costs = []
    start = time.time()

    t.backends.cudnn.benchmark = True
    best_xs_reconst = 100000 
    steps = 0
    for epoch in range(1, 3000 + 1):
        for iterno, (a_t, v_t, m_t, genre) in enumerate(va_train_loader):
            # get video, audio and beat data
            a_t = a_t.float().cuda()
            v_t = v_t.float().cuda() 
            m_t = m_t.float().cuda()
            genre = genre.float().cuda()


            # get output from encoder
            mx = mencoder(m_t)
            fuse_x = t.cat((mx, v_t), 2)
            xs_pred = encoder(fuse_x, genre, batch_size)


            
            with t.no_grad():
                xs_t, zs_t = vqvae._encode(a_t.transpose(1,2))
                level = 2 # 0, 1, 2 -> 2756, 689, 172
                ## pred output
                xs_code = []
                for l in range(3):
                    xs_code.append(xs_pred)
                zs_pred = vqvae.bottleneck.encode(xs_code)
                zs_pred_code = []
                zs_pred_code.append(zs_pred[level])
                xs_quantised_pred, audio_pred = vqvae._decode(zs_pred_code, start_level=2, end_level=3) # list
                gt_code = []
                gt_code.append(zs_t[level])
                xs_quantised_gt, gt_audio = vqvae._decode(gt_code, start_level=2, end_level=3)

            
            # calculate errors
            xs_error = F.l1_loss(xs_t[level].view(batch_size, 1, -1), xs_pred.view(batch_size, 1,-1))
            code_error = F.l1_loss(xs_quantised_gt[0].view(batch_size, 1, -1), xs_pred.view(batch_size, 1, -1))
            audio_error = F.l1_loss(a_t[:,:,0:44032].transpose(1,2), audio_pred)
            mel_t = fft(a_t)
            mel_pred = fft(audio_pred.transpose(1,2))
            mel_error = F.l1_loss(mel_t, mel_pred)

            
            # train discriminator
            xs_pred = xs_pred.view(batch_size,1, -1)
            xs_tmp = xs_t[level].view(batch_size,1, -1)
            #print("check pred input for Dis.", xs_pred.size(), t.min(xs_pred), t.max(xs_pred))
            #print("check gt input for Dis.", xs_tmp.size(), t.min(xs_tmp), t.max(xs_tmp))
            D_fake_det = netD(xs_pred.cuda().detach(), genre)
            D_real = netD(xs_tmp.cuda(), genre)
            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()
           
            if steps > -1:
                netD.zero_grad()
                loss_D.backward()
                optD.step()

            # train generator
            D_fake = netD(xs_pred.cuda(), genre)
            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            wt = D_weights * feat_weights
            for i in range(num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())
            
            mencoder.zero_grad()
            encoder.zero_grad()
            
            (loss_G + 3 * loss_feat + 5 * xs_error + 8 * code_error + 0 *audio_error + 15 * mel_error).backward()
            optG.step()




            # update tensorboard #
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), xs_error.item(), code_error.item(), audio_error.item(), mel_error.item()])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/xs_reconstruction", costs[-1][3], steps)
            writer.add_scalar("loss/codebook_reconstruction", costs[-1][4], steps)
            writer.add_scalar("loss/audio_reconstruction", costs[-1][5], steps)
            writer.add_scalar("loss/spec_reconstruction", costs[-1][6], steps)
            steps += 1

            if steps % 1000 == 0:
                st = time.time()
                syn_code_error = 0
                with t.no_grad():
                    for i, (v_t, a_t, m_t, genre) in enumerate(zip(test_video, test_audio, test_motion, test_genre)):
                        mx = mencoder(m_t)
                        fuse_x = t.cat((mx, v_t),2)
                        pred_xs = encoder(fuse_x, genre, batch_size)
                        xs_code = []
                        for j in range(3):
                            xs_code.append(pred_xs)
                        zs_pred = vqvae.bottleneck.encode(xs_code)
                        zs_pred_code = []
                        zs_pred_code.append(zs_pred[2])
                        _,pred_audio = vqvae._decode(zs_pred_code,start_level=2,end_level=3)
                        pred_audio = pred_audio.cpu().detach()#.numpy()
                        print("check syn_error for testing sample",i, t.min(pred_xs), t.max(pred_xs), t.min(zs_pred[2]), t.max(zs_pred[2]))
                        pred_audio = pred_audio.squeeze().detach().cpu().numpy()
                        sf.write("/home/zhuye/musicgen/samples/generated_%d.wav" % (i+1), pred_audio, 22050)

                t.save(mencoder.state_dict(), "/home/zhuye/musicgen/logs/mencoder.pt")
                t.save(encoder.state_dict(), "/home/zhuye/musicgen/logs/netG.pt")
                t.save(optG.state_dict(), "/home/zhuye/musicgen/logs/optG.pt")

                t.save(netD.state_dict(), "/home/zhuye/musicgen/logs/netD.pt")
                t.save(optD.state_dict(), "/home/zhuye/musicgen/logs/optD.pt")


                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)


            if steps % 100 == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(va_train_loader),
                        1000 * (time.time() - start) / 100,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()            



    exit()




def run(model, mode='ancestral', codes_file=None, audio_file=None, prompt_length_in_seconds=None, port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = Hyperparams(**kwargs)
    sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

    #with t.no_grad():
    train(model, device, hps, sample_hps)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fire.Fire(run)
