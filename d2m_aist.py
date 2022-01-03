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
from d2m.d2m_modules import vqEncoder_high,vqEncoder_low, Discriminator, motion_encoder, Audio2Mel
from d2m.utils import save_sample



def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--model", default='5b')
    parser.add_argument("--save_sample_path", required=True)
    parser.add_argument("--model_level", required=True)
    parser.add_argument("--log_path", default='./logs')

    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=32)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    args = parser.parse_args()
    return args


# Generate and save samples, alignment, and webpage for visualization.
def train(model, device, hps):

    args = parse_args()
    root = args.log_path
    batch_size = args.batch_size
    writer = SummaryWriter(str(root))
    save_sample_path = args.save_sample_path
    n_test_samples = args.n_test_samples
    model_level = args.model_level
    if model_level == "high":
        code_level = 2
        seq_len = 44032
        level_s = 2
        level_e = 3
    if model_level == "low":
        code_level = 1
        seq_len = 44096
        level_s = 1
        level_e = 2

    #### create the model ######
    num_D = args.num_D
    ndf = args.ndf
    n_layers_D = args.n_layers_D
    downsamp_factor = args.downsamp_factor

    vqvae= make_vae_model(model, device, hps).cuda()
    if model_level == "high":
        encoder = vqEncoder_high().cuda()
        vqvae.load_state_dict(t.load("./models/vqvae_high.pt"))
        vqvae.eval()
    if model_level == "low":
        encoder = vqEncoder_low().cuda()
        vqvae.load_state_dict(t.load("./models/vqvae_low.pt"))
        vqvae.eval()
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
    print("Finish creating the optimizer.")



    #### creat data loader ####
    va_train_set = VAMDataset( audio_files = './dataset/aist_audio_train_segment.txt', video_files = './dataset/aist_video_train_segment.txt', genre_label = './dataset/train_genre.npy', motion_files = './dataset/aist_motion_train_segment.txt')
    va_train_loader = DataLoader(va_train_set, batch_size = batch_size, num_workers=4, shuffle=True)
    va_test_set = VAMDataset( audio_files = './dataset/aist_audio_test_segment.txt', video_files = './dataset/aist_video_test_segment.txt', genre_label = './dataset/test_genre.npy', motion_files = './dataset/aist_motion_test_segment.txt', augment=False)
    va_test_loader = DataLoader(va_test_set, batch_size = 1)
    print("Finish data loader", len(va_train_loader), len(va_test_loader)) 
    

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
        gt_xs, zs_code = vqvae._encode(a_t.transpose(1,2))
        zs_middle = []
        zs_middle.append(zs_code[code_level])
        quantised_xs, out = vqvae._decode(zs_middle, start_level=level_s, end_level=level_e)
        audio = a_t.squeeze()
        out = out.squeeze()#.detach().cpu().numpy()
        gt_code_error = F.l1_loss(gt_xs[code_level], quantised_xs[0])
        audio_error = F.l1_loss(audio[0:seq_len], out)
        if not os.path.exists(save_sample_path):
            os.makedirs(save_sample_path)
        sample_original = 'original_' + str(i+1) + '.wav'
        sample_vqvae = 'vqvae_'+ str(i+1) + '.wav'
        sample_original = os.path.join(save_sample_path,sample_original)
        sample_vqvae = os.path.join(save_sample_path,sample_vqvae) 
        sf.write(sample_original, audio.detach().cpu().numpy(), 22050)   
        sf.write(sample_vqvae, out.detach().cpu().numpy(), 22050)

        if i > n_test_samples:
            break
    print("Finish dumping samples", len(test_audio), len(test_video))

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
                # level = 2 # 0, 1, 2 -> 2756, 689, 172
                ## pred output
                xs_code = []
                for l in range(3):
                    xs_code.append(xs_pred)
                zs_pred = vqvae.bottleneck.encode(xs_code)
                zs_pred_code = []
                zs_pred_code.append(zs_pred[code_level])
                xs_quantised_pred, audio_pred = vqvae._decode(zs_pred_code, start_level=level_s, end_level=level_e) # list
                gt_code = []
                gt_code.append(zs_t[code_level])
                xs_quantised_gt, gt_audio = vqvae._decode(gt_code, start_level=level_s, end_level=level_e)

            
            # calculate errors
            xs_error = F.l1_loss(xs_t[code_level].view(batch_size, 1, -1), xs_pred.view(batch_size, 1,-1))
            code_error = F.l1_loss(xs_quantised_gt[0].view(batch_size, 1, -1), xs_pred.view(batch_size, 1, -1))
            audio_error = F.l1_loss(a_t[:,:,0:seq_len].transpose(1,2), audio_pred)
            mel_t = fft(a_t)
            mel_pred = fft(audio_pred.transpose(1,2))
            mel_error = F.l1_loss(mel_t, mel_pred)

            
            # train discriminator
            xs_pred = xs_pred.view(batch_size,1, -1)
            xs_tmp = xs_t[code_level].view(batch_size,1, -1)
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
            
            (loss_G + 3 * loss_feat + 5 * xs_error + 8 * code_error + 40 *audio_error + 15 * mel_error).backward()
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
                        zs_pred_code.append(zs_pred[code_level])
                        _,pred_audio = vqvae._decode(zs_pred_code,start_level=level_s,end_level=level_e)
                        pred_audio = pred_audio.cpu().detach()#.numpy()
                        # print("testing sample",i, t.min(pred_xs), t.max(pred_xs), t.min(zs_pred[2]), t.max(zs_pred[2]))
                        pred_audio = pred_audio.squeeze().detach().cpu().numpy()
                        sample_generated = 'generated_'+ str(i+1) + '.wav'
                        sample_generated = os.path.join(save_sample_path,sample_generated)
                        sf.write(sample_generated, pred_audio, 22050)

                t.save(mencoder.state_dict(), "./logs/mencoder.pt")
                t.save(encoder.state_dict(), "./logs/netG.pt")
                t.save(optG.state_dict(), "./logs/optG.pt")

                t.save(netD.state_dict(), "./logs/netD.pt")
                t.save(optD.state_dict(), "./logs/optD.pt")


                print("Took %5.4fs to generate samples" % (time.time() - st), st)
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
    #sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))
    train(model, device, hps)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fire.Fire(run)
