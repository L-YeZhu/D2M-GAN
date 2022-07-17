import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize

from pathlib import Path
import numpy as np
import random
from PIL import Image


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        #random.seed(1234)
        #random.shuffle(self.audio_files)
        self.augment = augment

    def __getitem__(self, index):
        # Read audio
        #print('check audio files', self.audio_files, len(self.audio_files))
        #exit()
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        #print("check audio data", audio, audio.size(), sampling_rate)
        #print("check segment length", self.segment_length)
        #exit()
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0
        #print("check audio", audio, audio.size())
        #exit()
        return audio.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate




class VAMDataset(torch.utils.data.Dataset):
    """
    This is the main class to load video data.
    """        

    def __init__(self, audio_files, video_files, genre_label, motion_files, sampling_rate = 22050, augment = True):
        self.sampling_rate = sampling_rate
        self.segment_length = sampling_rate*2
        self.audio_files = files_to_list(audio_files)
        self.audio_files = [Path(audio_files).parent / x for x in self.audio_files]
        self.augment = augment
        self.video_files = files_to_list(video_files)
        self.video_files = [Path(video_files).parent / x for x in self.video_files]
        self.motion_files = files_to_list(motion_files)
        self.motion_files = [Path(motion_files).parent / x for x in self.motion_files]
        self.genre = np.load(genre_label)

    def __len__(self):
        return len(self.audio_files)


    def __getitem__(self, index):
        
        # Read audio
        audio_filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(audio_filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0


        # Read video
        video_filename = self.video_files[index]
        video = self.load_img_to_torch(video_filename)

        # Read motion
        motion_filename = self.motion_files[index]
        motion = self.load_motion_to_torch(motion_filename)

        # Read label
        #beat = self.labels[index]
        # read genre
        genre = self.genre[index]

        return audio.unsqueeze(0), video, motion, genre


    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate


    def load_img_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()

    def load_motion_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()



class TiktokDataset(torch.utils.data.Dataset):
    """
    This is the main class to load video data.
    """        

    def __init__(self, audio_files, video_files, motion_files, sampling_rate = 22050, augment = True):
        self.sampling_rate = sampling_rate
        self.segment_length = sampling_rate*2
        self.audio_files = files_to_list(audio_files)
        self.audio_files = [Path(audio_files).parent / x for x in self.audio_files]
        self.augment = augment
        self.video_files = files_to_list(video_files)
        self.video_files = [Path(video_files).parent / x for x in self.video_files]
        self.motion_files = files_to_list(motion_files)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        
        # Read audio
        audio_filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(audio_filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0


        # Read video
        video_filename = self.video_files[index]
        video = self.load_img_to_torch(video_filename)

        # Read motion
        motion_filename = self.motion_files[index]
        motion = self.load_motion_to_torch(motion_filename)


        return audio.unsqueeze(0), video, motion


    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate


    def load_img_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()

    def load_motion_to_torch(self, full_path):
        data = np.load(full_path)
        # print("motion data in the loader", np.shape(data))
        data = np.reshape(data, (60, 75))
        return torch.from_numpy(data).float()





class VADataset(torch.utils.data.Dataset):
    """
    This is the main class to load video data.
    """        

    def __init__(self, audio_files, video_files, sampling_rate = 22050, augment = True):
        self.sampling_rate = sampling_rate
        self.segment_length = sampling_rate*2
        self.audio_files = files_to_list(audio_files)
        self.audio_files = [Path(audio_files).parent / x for x in self.audio_files]
        self.augment = augment
        self.video_files = files_to_list(video_files)
        self.video_files = [Path(video_files).parent / x for x in self.video_files]
        #self.labels = np.load(beat_label)
        #self.genre = np.load(genre_label)

    def __len__(self):
        return len(self.audio_files)


    def __getitem__(self, index):
        # Read audio
        audio_filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(audio_filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0


        # Read video
        video_filename = self.video_files[index]
        video = self.load_img_to_torch(video_filename)

        return audio.unsqueeze(0), video


    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate


    def load_img_to_torch(self, full_path):
        #print("path check", full_path)
        data = np.load(full_path)
        return torch.from_numpy(data).float()





class VideoDataset(torch.utils.data.Dataset):
    """
    This is the main class to load video data.
    """        

    def __init__(self, training_files):
        self.video_files = files_to_list(training_files)
        self.video_files = [Path(training_files).parent / x for x in self.video_files]

    def load_img_to_torch(self, full_path):
        data = Image.open(full_path)
        return torch.from_numpy(data).float()
