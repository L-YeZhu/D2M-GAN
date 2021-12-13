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


class vqResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.ReflectionPad1d(dilation),
            nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = nn.Conv1d(dim, dim, kernel_size=1)

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

        return self.shortcut(x) + self.block(x)



class Encoder_high(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1024, 2058)
        # !!!! this is for ablation only
        #self.lin = nn.Linear(1024+256, 2058)
        #self.genre_embed = nn.Embedding(10,256)
        #self.fc = nn.Linear(2048, 2048)

        model = [
            nn.Conv1d(1, 32, kernel_size=6, stride=2, padding=1),
                ]
        model += [ResnetBlock(32, dilation=3**0)]
        model += [ResnetBlock(32, dilation=3**1)]
        model += [ResnetBlock(32, dilation=3**2)]
        model += [ResnetBlock(32, dilation=3**3)]
        model += [
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=41, stride=2, padding=1),
            ]
        model += [ResnetBlock(64, dilation=3**0)]
        model += [ResnetBlock(64, dilation=3**1)]
        model += [ResnetBlock(64, dilation=3**2)]
        model += [ResnetBlock(64, dilation=3**3)]
        model += [
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=41, stride=1, padding=1),
            ]
        model += [ResnetBlock(128, dilation=3**0)]
        model += [ResnetBlock(128, dilation=3**1)]
        model += [ResnetBlock(128, dilation=3**2)]
        model += [ResnetBlock(128, dilation=3**3)]
        model += [
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=41, stride=1, padding=1),
            ]
        model += [ResnetBlock(256, dilation=3**0)]
        model += [ResnetBlock(256, dilation=3**1)]
        model += [ResnetBlock(256, dilation=3**2)]
        model += [ResnetBlock(256, dilation=3**3)]
        model += [
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=41, stride=1, padding=1),
            ]
        model += [ResnetBlock(512, dilation=3**0)]
        model += [ResnetBlock(512, dilation=3**1)]
        model += [ResnetBlock(512, dilation=3**2)]
        model += [ResnetBlock(512, dilation=3**3)]
        model += [
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=41, stride=1, padding=1),
            #nn.Tanh(),
            ]
        model += [ResnetBlock(1024, dilation=3**0)]
        model += [ResnetBlock(1024, dilation=3**1)]
        model += [ResnetBlock(1024, dilation=3**2)]
        model += [
            nn.ReLU(),
            nn.Conv1d(1024, 2048, kernel_size=2, stride=1, padding=1)
                ]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = x.float()
        x = self.lin(x)
        out = self.model(x)
        return out






class vqEncoder_high(nn.Module):
    def __init__(self):
        super().__init__()
        ##self.lin = nn.Linear(1024+2048+256, 2058)
        #self.fc = nn.Linear(456, 344)
        # !!!! this is for ablation only
        self.lin = nn.Linear(2048+1024, 2058)
        #self.genre_embed = nn.Embedding(10,256)

        model = [
            nn.Conv1d(1, 32, kernel_size=6, stride=2, padding=1),
                ]
        model += [ResnetBlock(32, dilation=3**0)]
        model += [ResnetBlock(32, dilation=3**1)]
        model += [ResnetBlock(32, dilation=3**2)]
        model += [ResnetBlock(32, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=41, stride=2, padding=1),
            ]
        model += [ResnetBlock(64, dilation=3**0)]
        model += [ResnetBlock(64, dilation=3**1)]
        model += [ResnetBlock(64, dilation=3**2)]
        model += [ResnetBlock(64, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=41, stride=1, padding=1),
            ]
        model += [ResnetBlock(128, dilation=3**0)]
        model += [ResnetBlock(128, dilation=3**1)]
        model += [ResnetBlock(128, dilation=3**2)]
        model += [ResnetBlock(128, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=41, stride=1, padding=1),
            ]
        model += [ResnetBlock(256, dilation=3**0)]
        model += [ResnetBlock(256, dilation=3**1)]
        model += [ResnetBlock(256, dilation=3**2)]
        model += [ResnetBlock(256, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=41, stride=1, padding=1),
            ]
        model += [ResnetBlock(512, dilation=3**0)]
        model += [ResnetBlock(512, dilation=3**1)]
        model += [ResnetBlock(512, dilation=3**2)]
        model += [ResnetBlock(512, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2), 
            nn.Conv1d(512, 64, kernel_size=40, stride=1, padding=1),
            nn.Tanh(),
            ]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = x.float()
        # print("check input size", x.size())
        x = self.lin(x)
        out = self.model(x)
        return out * 100


class motion_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(75, 800)
        self.lin2 = nn.Linear(786, 1024)
        model = [
            nn.Conv1d(60, 256, kernel_size=6),
            nn.ReLU(),
                ]
        model += [ResnetBlock(256, dilation=3**0)]
        model += [ResnetBlock(256, dilation=3**1)]
        model += [ResnetBlock(256, dilation=3**2)]
        model += [ResnetBlock(256, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=4),
                ]
        model += [ResnetBlock(512, dilation=3**0)]
        model += [ResnetBlock(512, dilation=3**1)]
        model += [ResnetBlock(512, dilation=3**2)]
        model += [ResnetBlock(512, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1024, kernel_size=4),
                ]
        model += [ResnetBlock(1024, dilation=3**0)]
        model += [ResnetBlock(1024, dilation=3**1)]
        model += [ResnetBlock(1024, dilation=3**2)]
        model += [ResnetBlock(1024, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1, kernel_size=4),
                ]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        # print("check motion feature 0", x.size())
        x = self.lin1(x)
        #print("check motion feature 1", x.size())
        out = self.model(x)
        out = self.lin2(out)
        #print("check motion feature 2", out.size())
        return out






class vqEncoder_low(nn.Module):
    def __init__(self):
        super().__init__()
        #ratios = [8, 8, 2, 2]
        #self.hop_length = np.prod(ratios)
        #mult = int(2 ** len(ratios))
        #self.lin = nn.Linear(1024+2048+256,2048)
        #self.lin = nn.Linear(2048+256,2048)
        self.fc = nn.Linear(1045,1378)


        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(1, 32, kernel_size=6, stride=2, padding=1),
            #nn.LeakyReLU(0.2),
            #nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1),
        ]
        #for i in range(4):
        model += [ResnetBlock(32, dilation=3**0)]
        model += [ResnetBlock(32, dilation=3**1)]
        model += [ResnetBlock(32, dilation=3**2)]
        model += [ResnetBlock(32, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(32),
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=1),
        ]
        #for i in range(4):
        model += [ResnetBlock(64, dilation=3**0)]
        model += [ResnetBlock(64, dilation=3**1)]
        model += [ResnetBlock(64, dilation=3**2)]
        model += [ResnetBlock(64, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(64, 128, kernel_size=40, stride=2, padding=1),
        #    #nn.ReLU,
        ]
        #for i in range(4):
        model += [ResnetBlock(128, dilation=3**0)]
        model += [ResnetBlock(128, dilation=3**1)]
        model += [ResnetBlock(128, dilation=3**2)]
        model += [ResnetBlock(128, dilation=3**3)]

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(128, 256, kernel_size=40, stride=1, padding=1),
                ]
        model += [ResnetBlock(256, dilation=3**0)]
        model += [ResnetBlock(256, dilation=3**1)]
        model += [ResnetBlock(256, dilation=3**2)]
        model += [ResnetBlock(256, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(256, 512, kernel_size=40, stride=1, padding=1),
        ] 
        model += [ResnetBlock(512, dilation=3**0)]
        model += [ResnetBlock(512, dilation=3**1)]
        model += [ResnetBlock(512, dilation=3**2)]
        model += [ResnetBlock(512, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(512, 1024, kernel_size=40, stride=1, padding=1),
                ]
        model += [ResnetBlock(1024, dilation=3**0)]
        model += [ResnetBlock(1024, dilation=3**1)]
        model += [ResnetBlock(1024, dilation=3**2)]
        model += [ResnetBlock(1024, dilation=3**3)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(1024, 1024, kernel_size=40, stride=1, padding=1),
                ]
        model += [
            #nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(1024, 64, kernel_size=40, stride=1, padding=1),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = x.float()
        print("check input size", x.size())
        #x = self.lin(x)
        out = self.model(x)
        #out = out.view(batch_size, -1)
        out = 100*self.fc(out)
        #print("Generated output dim", out.size()) 
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
            nf = min(nf * stride, 1024)

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

    def forward(self, x):
        
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results

class NLayerDiscriminator_syn(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

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
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
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

class Discriminator_syn(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator_syn(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results
