import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConv2d(*arg, **kwargs):
    return weight_norm(nn.Conv2d(*arg, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def WNConvTranspose2d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose2d(*args, **kwargs))



class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        #print("check log_mel_spec size", log_mel_spec, log_mel_spec.size())
        #exit()
        return log_mel_spec


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios)) # multi = 16

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0), # input size = 80, ngf = 32
        ]
        #print("check input size", input_size)

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        #print("check x in generator", x.size()) #(16,80,32)
        #print("check x after layer3&4", self.model(x).size())
        #exit()
        return self.model(x)


class VAResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            #nn.ReLU(),
            nn.ReflectionPad2d(dilation),
            WNConv2d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            #nn.ReLU(),
            WNConv2d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        #return self.shortcut(x)
        #return self.block(x)
        return self.shortcut(x) + self.block(x)



class vqEncoder_middle(nn.Module):
    def __init__(self):
        super().__init__()
        #ratios = [8, 8, 2, 2]
        #self.hop_length = np.prod(ratios)
        #mult = int(2 ** len(ratios))
        self.lin = nn.Linear(1006,689)
        #self.genre = genre
        #self.beat = beat
        #self.genre_embed = nn.Embedding(10,256)


        model = [
            #nn.ReflectionPad2d(3),
            WNConv1d(2, 64, kernel_size=8, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            #nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1),
        ]
        model += [ResnetBlock(64)]
        model += [ResnetBlock(64)]
        model += [ResnetBlock(64)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(64, 256, kernel_size=16, stride=1, padding=1),
        ##    #nn.ReLU(),
        ]
        model += [ResnetBlock(256, dilation=3**0)]
        model += [ResnetBlock(256, dilation=3**1)]
        model += [ResnetBlock(256, dilation=3**2)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(256, 512, kernel_size=16, stride=1, padding=1),
        #    #nn.ReLU,
        ]
        model += [ResnetBlock(512, dilation=3**0)]
        model += [ResnetBlock(512, dilation=3**1)]
        model += [ResnetBlock(512, dilation=3**2)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(512, 64, kernel_size=32, stride=1, padding=1),
        ] 
        model += [ResnetBlock(64, dilation=3**0)]
        model += [ResnetBlock(64, dilation=3**1)]
        model += [ResnetBlock(64, dilation=3**2)]
        model += [
                nn.LeakyReLU(0.2),
                #nn.AvgPool2d(kernel_size=3),
                WNConv1d(64, 64, kernel_size=3, stride=2, padding=1),
                #nn.ReLU(),
                #nn.Tanh(),
                #self.lin(),
                ]
        model += [ResnetBlock(64)]
        model += [ResnetBlock(64)]
        model += [ResnetBlock(64)]
        model += [
                nn.LeakyReLU(0.2),
                WNConv1d(64, 64, kernel_size=5, stride=1, padding=1),
                nn.Tanh(),
                ]
        self.model = nn.Sequential(*model)
        #self.lin = nn.Sequential(lin)
        self.apply(weights_init)

    def forward(self, x, genre):
        x = x.float()
        #print("check initial input", x.size(), genre.size())
        label_embed = genre.unsqueeze(1).repeat([1,2,1])
        #print("check size", label_embed.size())
        x = torch.cat((x, label_embed),2)
        #print("check input size after label condition", x.size())
        out = self.model(x)
        #print("Generated output dim", out.size()) 
        #out = torch.flatten(out, start_dim=2)
        out = self.lin(out)
        #print("Generates output dim after lin", out.size())
        return out 





class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()
        
        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=16),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU(),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 512)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
                #nn.ReLU(),
            )

        nf = min(nf * 2, 512)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU(),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        #self.genre_embed = nn.Embedding(10,689)
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x, genre):
        #print("check input for D", x.size(), genre.size())
        #label_embed = genre.unsqueeze(1).repeat([1,2,1])
        #x = torch.cat((x,label_embed),2)
        
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
            #print("check output in D", x.size(), len(results))
        return results
