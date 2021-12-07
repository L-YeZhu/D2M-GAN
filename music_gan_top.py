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
from jukebox.data.labels import EmptyLabeller
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav, load_audio
from jukebox.make_models import make_vae_model
from jukebox.align import get_alignment
from jukebox.save_html import save_html
from jukebox.utils.sample_utils import split_batch, get_starts
from jukebox.utils.dist_utils import print_once
import fire
import librosa
import soundfile as sf
import torchlibrosa as tl

from v2vq.dataset import VAMDataset
from v2vq.v2vq_modules import vqEncoder_top, Discriminator, motion_encoder, Audio2Mel
from v2vq.utils import save_sample



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



## Sample a partial window of length<n_ctx with tokens_to_sample new tokens on level=level
#def sample_partial_window(zs, labels, sampling_kwargs, level, prior, tokens_to_sample, hps):
#    z = zs[level]
#    n_ctx = prior.n_ctx
#    current_tokens = z.shape[1]
#    if current_tokens < n_ctx - tokens_to_sample:
#        sampling_kwargs['sample_tokens'] = current_tokens + tokens_to_sample
#        start = 0
#    else:
#        sampling_kwargs['sample_tokens'] = n_ctx
#        start = current_tokens - n_ctx + tokens_to_sample
#
#    return sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)
#
## Sample a single window of length=n_ctx at position=start on level=level
#def sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps):
#    n_samples = hps.n_samples
#    n_ctx = prior.n_ctx
#    end = start + n_ctx
#
#    # get z already sampled at current level
#    z = zs[level][:,start:end]
#
#    if 'sample_tokens' in sampling_kwargs:
#        # Support sampling a window shorter than n_ctx
#        sample_tokens = sampling_kwargs['sample_tokens']
#    else:
#        sample_tokens = (end - start)
#    conditioning_tokens, new_tokens = z.shape[1], sample_tokens - z.shape[1]
#
#    print_once(f"Sampling {sample_tokens} tokens for [{start},{start+sample_tokens}]. Conditioning on {conditioning_tokens} tokens")
#
#    if new_tokens <= 0:
#        # Nothing new to sample
#        return zs
#    
#    # get z_conds from level above
#    z_conds = prior.get_z_conds(zs, start, end)
#
#    # set y offset, sample_length and lyrics tokens
#    y = prior.get_y(labels, start)
#
#    empty_cache()
#
#    max_batch_size = sampling_kwargs['max_batch_size']
#    del sampling_kwargs['max_batch_size']
#
#
#    z_list = split_batch(z, n_samples, max_batch_size)
#    z_conds_list = split_batch(z_conds, n_samples, max_batch_size)
#    y_list = split_batch(y, n_samples, max_batch_size)
#    z_samples = []
#    for z_i, z_conds_i, y_i in zip(z_list, z_conds_list, y_list):
#        z_samples_i = prior.sample(n_samples=z_i.shape[0], z=z_i, z_conds=z_conds_i, y=y_i, **sampling_kwargs)
#        z_samples.append(z_samples_i)
#    z = t.cat(z_samples, dim=0)
#
#    sampling_kwargs['max_batch_size'] = max_batch_size
#
#    # Update z with new sample
#    z_new = z[:,-new_tokens:]
#    zs[level] = t.cat([zs[level], z_new], dim=1)
#    return zs
#
## Sample total_length tokens at level=level with hop_length=hop_length
#def sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):
#    print_once(f"Sampling level {level}")
#    if total_length >= prior.n_ctx:
#        for start in get_starts(total_length, prior.n_ctx, hop_length):
#            zs = sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)
#    else:
#        zs = sample_partial_window(zs, labels, sampling_kwargs, level, prior, total_length, hps)
#    return zs
#
## Sample multiple levels
#def _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps):
#    alignments = None
#    for level in reversed(sample_levels):
#        prior = priors[level]
#        prior.cuda()
#        empty_cache()
#
#        # Set correct total_length, hop_length, labels and sampling_kwargs for level
#        assert hps.sample_length % prior.raw_to_tokens == 0, f"Expected sample_length {hps.sample_length} to be multiple of {prior.raw_to_tokens}"
#        total_length = hps.sample_length//prior.raw_to_tokens
#        hop_length = int(hps.hop_fraction[level]*prior.n_ctx)
#        zs = sample_level(zs, labels[level], sampling_kwargs[level], level, prior, total_length, hop_length, hps)
#
#        prior.cpu()
#        empty_cache()
#
#        # Decode sample
#        x = prior.decode(zs[level:], start_level=level, bs_chunks=zs[level].shape[0])
#
#        if dist.get_world_size() > 1:
#            logdir = f"{hps.name}_rank_{dist.get_rank()}/level_{level}"
#        else:
#            logdir = f"{hps.name}/level_{level}"
#        if not os.path.exists(logdir):
#            os.makedirs(logdir)
#        t.save(dict(zs=zs, labels=labels, sampling_kwargs=sampling_kwargs, x=x), f"{logdir}/data.pth.tar")
#        save_wav(logdir, x, hps.sr)
#        if alignments is None and priors[-1] is not None and priors[-1].n_tokens > 0 and not isinstance(priors[-1].labeller, EmptyLabeller):
#            alignments = get_alignment(x, zs, labels[-1], priors[-1], sampling_kwargs[-1]['fp16'], hps)
#        save_html(logdir, x, zs, labels[-1], alignments, hps)
#    return zs
#
## Generate ancestral samples given a list of artists and genres
#def ancestral_sample(labels, sampling_kwargs, priors, hps):
#    sample_levels = list(range(len(priors)))
#    print("*****Ancestral sampling check*****")
#    print(labels, hps)
#    zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
#    print("zs before sampling", zs,len(zs), zs[0].size())
#    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
#    print("zs after sampling", zs, len(zs), zs[0].size(),zs[1].size(),zs[2].size())
#    return zs
#
## Continue ancestral sampling from previously saved codes
#def continue_sample(zs, labels, sampling_kwargs, priors, hps):
#    sample_levels = list(range(len(priors)))
#    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
#    return zs
#
## Upsample given already generated upper-level codes
#def upsample(zs, labels, sampling_kwargs, priors, hps):
#    sample_levels = list(range(len(priors) - 1))
#    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
#    return zs
#
## Prompt the model with raw audio input (dimension: NTC) and generate continuations
#def primed_sample(x, labels, sampling_kwargs, priors, hps):
#    sample_levels = list(range(len(priors)))
#    zs = priors[-1].encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
#    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
#    return zs
#
## Load `duration` seconds of the given audio files to use as prompts
#def load_prompts(audio_files, duration, hps):
#    xs = []
#    for audio_file in audio_files:
#        x = load_audio(audio_file, sr=hps.sr, duration=duration, offset=0.0, mono=True)
#        x = x.T # CT -> TC
#        xs.append(x)
#    while len(xs) < hps.n_samples:
#        xs.extend(xs)
#    xs = xs[:hps.n_samples]
#    x = t.stack([t.from_numpy(x) for x in xs])
#    x = x.to('cuda', non_blocking=True)
#    return x


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
    #tanh = activation().cuda()
    #fft = Audio2Mel(n_mel_channels=args.n_mel_channels).cuda() 
    
    print(mencoder)
    print(encoder)
    print(netD)
    #exit()
    #print(vqvae)


    
    #### create optimizer #####
    t_param = list(mencoder.parameters()) + list(encoder.parameters())
    optG = t.optim.Adam(t_param, lr=1e-4, betas=(0.5, 0.9))
    optD = t.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    vqvae.load_state_dict(t.load("/data/zhuye/music_results/d2m_models/logs_top/top_vqvae1.pt"))
    #optM = t.optim.Adam(mencoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
    vqvae.eval()

    #### continue training ####
    #load_root = '/home/zhuye/musicgen/logs'
    #mencoder.load_state_dict(t.load("/home/zhuye/musicgen/logs_top/mencoder.pt"))
    #encoder.load_state_dict(t.load("/home/zhuye/musicgen/logs_top/netG.pt"))
    #optG.load_state_dict(t.load("/home/zhuye/musicgen/logs_1017/optG.pt"))
    #netD.load_state_dict(t.load("/home/zhuye/musicgen/logs_1017/netD.pt"))
    #optD.load_state_dict(t.load("/home/zhuye/musicgen/logs_1017/optD.pt"))
    print("Now continue training...")
    #exit()


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
        #save_sample(root / ("original_%d.wav" % i), 22050, audio)
        #writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=22050)
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
            v_t = v_t.float().cuda() # nhwc -> ncwh
            m_t = m_t.float().cuda()
            genre = genre.float().cuda()
            #beat = beat.cuda()
            #print("check data from loader", a_t.size(), v_t.size(), genre.size())

            # get output from encoder
            mx = mencoder(m_t)
            fuse_x = t.cat((mx, v_t), 2)
            xs_pred = encoder(fuse_x, genre, batch_size)

            ## !!!!! this is for ablation

            #xs_pred = encoder(fuse_x, genre, batch_size)
            print("check xs", xs_pred.size())
            exit()
            #mel_pred = xs_pred / 100
            #print("check mel_pred 1", mel_pred.size(), t.min(mel_pred), t.max(mel_pred))
            #exit()
            #mel_pred = fft(mel_pred)
            #print("check mel_pred 2", mel_pred.size())

            #print("check xs_pred", xs_pred, t.min(xs_pred), t.max(xs_pred))
            #xs_pred = 100 * xs_pred
            #print("check pred after scale", xs_pred, t.min(xs_pred), t.max(xs_pred))
            #exit()
            #print("check output from encoder", xs_pred.size(), xs_pred.requires_grad)
            
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
                #print("check output from two decoder", audio_pred - temp_audio_pred, xs_quantised_pred[0].size())
                ## gt output
                gt_code = []
                gt_code.append(zs_t[level])
                #print("check the gt codebook", zs_t[level].size(), zs_t[level])
                #gt_audio = vqvae.decode(gt_code, start_level=1, end_level=2)
                xs_quantised_gt, gt_audio = vqvae._decode(gt_code, start_level=2, end_level=3)
                #print("check gt output from two decoder",  t.min(xs_quantised_gt[0]), t.max(xs_quantised_gt[0]))

            
            # calculate errors
            xs_error = F.l1_loss(xs_t[level].view(batch_size, 1, -1), xs_pred.view(batch_size, 1,-1))
            code_error = F.l1_loss(xs_quantised_gt[0].view(batch_size, 1, -1), xs_pred.view(batch_size, 1, -1))
            audio_error = F.l1_loss(a_t[:,:,0:44032].transpose(1,2), audio_pred)
            mel_t = fft(a_t)
            #print("size check", a_t.size(), audio_pred.size())
            mel_pred = fft(audio_pred.transpose(1,2))
            mel_error = F.l1_loss(mel_t, mel_pred)
            #stft = tl.Spectrogram(n_fft=2048, hop_length=512).cuda()
            #mel = tl.LogmelFilterBank(sr=22050, n_fft=2048, n_mels=64).cuda()
            #sp_t = stft(a_t.squeeze())
            #sp_pred = stft(audio_pred.squeeze())
            #spec_t = mel(sp_t)#.cpu().detach().numpy())
            #spec_pred = mel(sp_pred)#.cpu().detach().numpy())
            #spec_error = F.l1_loss(spec_t, spec_pred)# + F.l1_loss(spec_t[1].squeeze(), spec_pred[1].squeeze())
            
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
            #if steps > 5000:
            #    if epoch < 100:
            #        (loss_G +  5 * loss_feat + 0 * xs_error + 10 * code_error + 10 * audio_error + 10 * mel_error).backward()
            #    else:
            #        (loss_G + 10 * loss_feat + 1 * code_error + 10 * audio_error + 10 * mel_error).backward()
            #else:
            #    (10 * code_error + 20 * audio_error + 10 * mel_error).backward()
            optG.step()


            #if epoch > 200:
            #    optS = t.optim.Adam(vqave.parameters(), lr=1e-6, betas=(0.5, 0.9))
            #    vqvae.zero_grad()
            #    (5 * audio_error + 5 * mel_error).backward()
            #    optS.step()


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
                        #if i < 100:
                        #    xs_code = []
                        #    gt_xs, zs = vqvae._encode(a_t.transpose(1,2))
                        #    zs_m = []
                        #    zs_m.append(zs[2])
                        #    _,out = vqvae._decode(zs_m, start_level=2, end_level=3)
                        #    out = out.squeeze().detach().cpu().numpy()
                        #    #print("check gt data", gt_xs[1], zs[1], t.min(gt_xs[1]), t.max(gt_xs[1]))
                        #    sf.write('/home/zhuye/musicgen/samples/test_%d.wav'% (i+1), out, 22050 )
                        mx = mencoder(m_t)
                        fuse_x = t.cat((mx, v_t),2)
                        pred_xs = encoder(fuse_x, genre, batch_size)
                        #pred_xs = 100 * pred_xs
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

                        ### Some tricks here
                        #xs_code_test = []
                        #xs_code_test.append(pred_xs*2)
                        #zs_pred_test = vqvae.bottleneck.encode(xs_code_test)
                        #_,pred_audio_test = vqvae._decode(zs_pred_test, start_level=2, end_level=3)
                        #pred_audio_test = pred_audio_test.squeeze().cpu().detach().numpy()
                        #sf.write("/home/zhuye/musicgen/samples/test_%d.wav" % (i+1), pred_audio_test, 22050)



                        #save_sample(root / ("generated_%d.wav" % i), 22050, pred_audio)
                        #writer.add_audio(
                        #    "generated/sample_%d.wav" % i,
                        #     pred_audio,
                        #     epoch,
                        #     sample_rate=22050,
                        #)
                t.save(mencoder.state_dict(), "/home/zhuye/musicgen/logs/mencoder.pt")
                t.save(encoder.state_dict(), "/home/zhuye/musicgen/logs/netG.pt")
                t.save(optG.state_dict(), "/home/zhuye/musicgen/logs/optG.pt")

                t.save(netD.state_dict(), "/home/zhuye/musicgen/logs/netD.pt")
                t.save(optD.state_dict(), "/home/zhuye/musicgen/logs/optD.pt")

                #if np.asarray(costs).mean(0)[-2] < best_xs_reconst:
                #    best_xs_reconst = np.asarray(costs).mean(0)[-1]
                #    t.save(netD.state_dict(), "/home/zhuye/musicgen/logs/best_netD.pt")
                #    t.save(encoder.state_dict(),"/home/zhuye/musicgen/logs/best_netG.pt")

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

            #exit()



    exit()







    #assert hps.sample_length//priors[-2].raw_to_tokens >= priors[-2].n_ctx, f"Upsampling needs atleast one ctx in get_z_conds. Please choose a longer sample length"

    #total_length = hps.total_sample_length_in_seconds * hps.sr
    #offset = 0

    ## Set artist/genre/lyrics for your samples here!
    ## We used different label sets in our models, but you can write the human friendly names here and we'll map them under the hood for each model.
    ## For the 5b/5b_lyrics model and the upsamplers, labeller will look up artist and genres in v2 set. (after lowercasing, removing non-alphanumerics and collapsing whitespaces to _).
    ## For the 1b_lyrics top level, labeller will look up artist and genres in v3 set (after lowercasing).
    #metas = [dict(artist = "Alan Jackson",
    #              genre = "Country",
    #              lyrics = poems['ozymandias'],
    #              total_length=total_length,
    #              offset=offset,
    #              ),
    #         dict(artist="Joe Bonamassa",
    #              genre="Blues Rock",
    #              lyrics=gpt_2_lyrics['hottub'],
    #              total_length=total_length,
    #              offset=offset,
    #              ),
    #         dict(artist="Frank Sinatra",
    #              genre="Classic Pop",
    #              lyrics=gpt_2_lyrics['alone'],
    #              total_length=total_length,
    #              offset=offset,
    #              ),
    #         dict(artist="Ella Fitzgerald",
    #              genre="Jazz",
    #              lyrics=gpt_2_lyrics['count'],
    #              total_length=total_length,
    #              offset=offset,
    #              ),
    #         dict(artist="Céline Dion",
    #              genre="Pop",
    #              lyrics=gpt_2_lyrics['darkness'],
    #              total_length=total_length,
    #              offset=offset,
    #              ),
    #         ]
    #while len(metas) < hps.n_samples:
    #    metas.extend(metas)
    #metas = metas[:hps.n_samples]

    #labels = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in priors]
    #for label in labels:
    #    assert label['y'].shape[0] == hps.n_samples

    #lower_level_chunk_size = 32
    #lower_level_max_batch_size = 16
    #if model == '1b_lyrics':
    #    chunk_size = 32
    #    max_batch_size = 16
    #else:
    #    chunk_size = 16
    #    max_batch_size = 3
    #sampling_kwargs = [dict(temp=0.99, fp16=True, chunk_size=lower_level_chunk_size, max_batch_size=lower_level_max_batch_size),
    #                   dict(temp=0.99, fp16=True, chunk_size=lower_level_chunk_size, max_batch_size=lower_level_max_batch_size),
    #                   dict(temp=0.99, fp16=True, chunk_size=chunk_size, max_batch_size=max_batch_size)]

    #if sample_hps.mode == 'ancestral':
    #    ancestral_sample(labels, sampling_kwargs, priors, hps)
    #elif sample_hps.mode in ['continue', 'upsample']:
    #    assert sample_hps.codes_file is not None
    #    top_raw_to_tokens = priors[-1].raw_to_tokens
    #    if sample_hps.prompt_length_in_seconds is not None:
    #        duration = (int(sample_hps.prompt_length_in_seconds * hps.sr) // top_raw_to_tokens) * top_raw_to_tokens
    #    else:
    #        duration = None
    #    zs = load_codes(sample_hps.codes_file, duration, priors, hps)
    #    if sample_hps.mode == 'continue':
    #        continue_sample(zs, labels, sampling_kwargs, priors, hps)
    #    elif sample_hps.mode == 'upsample':
    #        upsample(zs, labels, sampling_kwargs, priors, hps)
    #elif sample_hps.mode == 'primed':
    #    assert sample_hps.audio_file is not None
    #    assert sample_hps.prompt_length_in_seconds is not None
    #    audio_files = sample_hps.audio_file.split(',')
    #    top_raw_to_tokens = priors[-1].raw_to_tokens
    #    duration = (int(sample_hps.prompt_length_in_seconds * hps.sr) // top_raw_to_tokens) * top_raw_to_tokens
    #    x = load_prompts(audio_files, duration, hps)
    #    primed_sample(x, labels, sampling_kwargs, priors, hps)
    #else:
    #    raise ValueError(f'Unknown sample mode {sample_hps.mode}.')


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
