import torch
import numpy as np
from pypianoroll import Multitrack, BinaryTrack

# Data
n_tracks = 1  # 사용되는 트랙의 수. 데이터 전처리 시 이 트랙 수는 모두 동일해야함.
n_pitches = 72  # number of pitches
lowest_pitch = 24  # MIDI note number of the lowest pitch
n_samples_per_song = 8  # number of samples to extract from each song in the datset
n_measures = 4  # number of measures per sample
beat_resolution = 4  # temporal resolution of a beat (in timestep)
programs = [0]  # program number for each track
is_drums = [False]  # drum indicator for each track
track_names = ['Track 1']  # 각 트랙의 이름. 반드시 맞춰야 할 필요는 없음
tempo = 100

# Training
batch_size = 64
latent_dim = 1024
n_steps = 50000

# Sampling
sample_interval = 1000  # interval to run the sampler (in step)
n_samples = 4

measure_resolution = 4 * beat_resolution
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)

class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)

class Generator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneraterBlock(latent_dim, 512, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(512, 256, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(256, 128, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(128, 64, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList([
            GeneraterBlock(64, 32, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.transconv5 = torch.nn.ModuleList([
            GeneraterBlock(32, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = [transconv(x) for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x