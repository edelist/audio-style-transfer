"""
Mel spectrogram to audio 

from: https://www.kaggle.com/code/gaurav41/how-to-convert-audio-to-mel-spectrogram-to-audio/notebook

model weights: https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth?dl=0

"""
import os
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from wavenet_vocoder import builder

import random
import struct
from pathlib import Path
from typing import Optional, Union

import librosa
import webrtcvad
# from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import binary_dilation

import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
import pickle

from audio_utils import *


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


class DictWithDotNotation(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DictWithDotNotation(value)
            self[key] = value


class GetDictWithDotNotation(DictWithDotNotation):

    def __init__(self, hp_dict):
        super(DictWithDotNotation, self).__init__()

        hp_dotdict = DictWithDotNotation(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = DictWithDotNotation.__getitem__
    __setattr__ = DictWithDotNotation.__setitem__
    __delattr__ = DictWithDotNotation.__delitem__

PROJECT_DIR = ""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device type available = '{device}'")

hparam_dict = {
    # general parameters
    'general':{
        # small error
        'small_err': 1e-6,
        'is_training_mode': True,
        'device': device,
        'project_root': PROJECT_DIR,
    },
    # path to raw audio file
    "raw_audio": {
        "raw_audio_path": "static/raw_data/wavs",
        "train_spectrogram_path": "static/spectrograms/train",
        "test_spectrogram_path": "static/spectrograms/test",
        "train_percent": .8,
    },
    # audio --> same audio settings to be used in wavenet model to
    # reconstruct audio from mel-spectrogram
    "audio": {
        "sampling_rate": 16000,
        # "sampling_rate": 22500,
        # Number of spectrogram frames in a partial utterance
        "partials_n_frames": 180,  # 1600 ms

        "n_fft": 1024,  # 1024 seems to work well
        "hop_length": 1024 // 4,  # n_fft/4 seems to work better

        "mel_window_length": 25,  # In milliseconds
        "mel_window_step": 10,  # In milliseconds
        "mel_n_channels": 80,

    },
    ## Voice Activation Detection
    "vad": {
        # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
        # This sets the granularity of the VAD. Should not need to be changed.
        "vad_window_length": 30,  # In milliseconds
        # Number of frames to average together when performing the moving average smoothing.
        # The larger this value, the larger the VAD variations must be to not get smoothed out.
        "vad_moving_average_width": 8,
        # Maximum number of consecutive silent frames a segment can have.
        "vad_max_silence_length": 6,

        ## Audio volume normalization
        "audio_norm_target_dBFS": -30,
        "rate_partial_slices": 1.3,
        "min_coverage": 0.75,
    },
    "m_wave_net": {
        "gen": {
            "best_model_path": "static/model_chk_pts/wavenet_model/checkpoint_step001000000_ema.pth"
        },
        "hp": {
            # DO NOT CHANGE THESE HP
            'name': "wavenet_vocoder",

            # Convenient model builder
            'builder': "wavenet",

            # Input type:
            # 1. raw [-1, 1]
            # 2. mulaw [-1, 1]
            # 3. mulaw-quantize [0, mu]
            # If input_type is raw or mulaw, network assumes scalar input and
            # discretized mixture of logistic distributions output, otherwise one-hot
            # input and softmax output are assumed.
            # **NOTE**: if you change the one of the two parameters below, you need to
            # re-run preprocessing before training.
            'input_type': "raw",
            'quantize_channels': 65536,  # 65536 or 256

            # Audio: these 4 items to be same as used to create mel out of audio
            # commented back in sr, fft_size, hop_size, num_mels
            'sample_rate': 16000,
            # 'sample_rate': 22500,
            'fft_size': 1024,
            # # shift can be specified by either hop_size or frame_shift_ms
            'hop_size': 256,
            'num_mels': 80,

            # this is only valid for mulaw is True
            'silence_threshold': 2,

            'fmin': 125,
            'fmax': 7600,
            'frame_shift_ms': None,
            'min_level_db': -100,
            'ref_level_db': 20,
            # whether to rescale waveform or not.
            # Let x is an input waveform, rescaled waveform y is given by:
            # y = x / np.abs(x).max() * rescaling_max
            'rescaling': True,
            'rescaling_max': 0.999,
            # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
            # happen depends on min_level_db and ref_level_db, causing clipping noise.
            # If False, assertion is added to ensure no clipping happens.o0
            'allow_clipping_in_normalization': True,

            # Mixture of logistic distributions:
            'log_scale_min': float(-32.23619130191664),

            # Model:
            # This should equal to `quantize_channels` if mu-law quantize enabled
            # otherwise num_mixture * 3 (pi, mean, log_scale)
            'out_channels': 10 * 3,
            'layers': 24,
            'stacks': 4,
            'residual_channels': 512,
            'gate_channels': 512,  # split into 2 gropus internally for gated activation
            'skip_out_channels': 256,
            'dropout': 1 - 0.95,
            'kernel_size': 3,
            # If True, apply weight normalization as same as DeepVoice3
            'weight_normalization': True,
            # Use legacy code or not. Default is True since we already provided a model
            # based on the legacy code that can generate high-quality audio.
            # Ref: https://github.com/r9y9/wavenet_vocoder/pull/73
            'legacy': True,

            # Local conditioning (set negative value to disable))
            'cin_channels': 80,
            # If True, use transposed convolutions to upsample conditional features,
            # otherwise repeat features to adjust time resolution
            'upsample_conditional_features': True,
            # should np.prod(upsample_scales) == hop_size
            'upsample_scales': [4, 4, 4, 4],
            # Freq axis kernel size for upsampling network
            'freq_axis_kernel_size': 3,

            # Global conditioning (set negative value to disable)
            # currently limited for speaker embedding
            # this should only be enabled for multi-speaker dataset
            'gin_channels': -1,  # i.e., speaker embedding dim
            'n_speakers': -1,

            # Data loader
            'pin_memory': True,
            'num_workers': 2,

            # train/test
            # test size can be specified as portion or num samples
            'test_size': 0.0441,  # 50 for CMU ARCTIC single speaker
            'test_num_samples': None,
            'random_state': 1234,

            # Loss

            # Training:
            'batch_size': 2,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_eps': 1e-8,
            'amsgrad': False,
            'initial_learning_rate': 1e-3,
            # see lrschedule.py for available lr_schedule
            'lr_schedule': "noam_learning_rate_decay",
            'lr_schedule_kwargs': {},  # {"anneal_rate": 0.5, "anneal_interval": 50000},
            'nepochs': 2000,
            'weight_decay': 0.0,
            'clip_thresh': -1,
            # max time steps can either be specified as sec or steps
            # if both are None, then full audio samples are used in a batch
            'max_time_sec': None,
            'max_time_steps': 8000,
            # Hold moving averaged parameters and use them for evaluation
            'exponential_moving_average': True,
            # averaged = decay * averaged + (1 - decay) * x
            'ema_decay': 0.9999,

            # Save
            # per-step intervals
            'checkpoint_interval': 10000,
            'train_eval_interval': 10000,
            # per-epoch interval
            'test_eval_epoch_interval': 5,
            'save_optimizer_state': True,

            # Eval:
        }
    }
}

# this hp will be used throughout the project
hp = GetDictWithDotNotation(hparam_dict)

# few calculated values from wavenet model
hp.m_wave_net.hp.sample_rate = hp.audio.sampling_rate
hp.m_wave_net.hp.fft_size = hp.audio.n_fft
hp.m_wave_net.hp.hop_size = hp.audio.hop_length
hp.m_wave_net.hp.num_mels = hp.audio.mel_n_channels

def pySTFT(x, fft_length=1024, hop_length=256):
    """
    this function returns spectrogram (short time fourier transform)
    :param x: np array for the audio file
    :param fft_length: fft length for fast fourier transform (https://www.youtube.com/watch?v=E8HeD-MUrjY)
    :param hop_length: hop_length is the sliding overlapping window size normally fft//4 works the best
    :return: spectrogram in the form of np array
    """
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # window of given type and length
    fft_window = signal.get_window('hann', fft_length, fftbins=True)
    # compute 1-dim discrete Fourier transform for real input
    # result = np.fft.rfft(fft_window * result, n=fft_length).T

    result = librosa.core.stft(x, n_fft=1024, hop_length=hop_length)

    return np.abs(result)

def butter_highpass(cutoff, fs, order=5):
    """
    high pass Butterworth digital filter
    Params: 
    - cutoff = cutoff freq of filter, frequencies below this value will be attenuated
    - fs = sampling frequency of signal
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq    # normalized cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    # b, a = numerator, denominator polynomials of IIR filter (infinite impulse response)
    return b, a

def wav_to_mel_spectrogram(wav, hp):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """

    # creating mel basis matrix: linear transform matrix
    # to project FFT bins onto Mel-frequency bins
    mel_basis = mel(sr=hp.audio.sampling_rate,  # sampling rate of signal
                    n_fft=hp.audio.n_fft,       # num FFT components
                    fmin=90,                    # lowest frequency (Hz) def: 90
                    fmax=7600,                  # highest frequency (Hz) def: 7600
                    n_mels=hp.audio.mel_n_channels).T   # num Mel bands to generate

    min_level = np.exp(-100 / 20 * np.log(10))

    # getting audio as a np array
    pp_wav = preprocess_wav(wav, hp, source_sr=22500)

    # Compute spectrogram
    spectrogram = pySTFT(pp_wav).T
    # Convert to mel and normalize
    mel_spect = np.dot(spectrogram, mel_basis)
    d_db = 20 * np.log10(np.maximum(min_level, mel_spect)) - 16
    norm_mel_spect = np.clip((d_db + 100) / 100, 0, 1)

    return norm_mel_spect

def shuffle_along_axis(a, axis):
    """
    :param a: nd array e.g. [40, 180, 80]
    :param axis: array axis. e.g. 0
    :return: a shuffled np array along the given axis
    """
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


### Build Wavenet model - used to convert mel-spectrogram to audio
torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


hparams = hp.m_wave_net.hp

def build_model():
    """Build Wavenet model"""
    model = getattr(builder, hparams.builder)(
    out_channels=hparams.out_channels,
    layers=hparams.layers,
    stacks=hparams.stacks,
    residual_channels=hparams.residual_channels,
    gate_channels=hparams.gate_channels,
    skip_out_channels=hparams.skip_out_channels,
    cin_channels=hparams.cin_channels,
    gin_channels=hparams.gin_channels,
    weight_normalization=hparams.weight_normalization,
    n_speakers=hparams.n_speakers,
    dropout=hparams.dropout,
    kernel_size=hparams.kernel_size,
    upsample_conditional_features=hparams.upsample_conditional_features,
    upsample_scales=hparams.upsample_scales,
    freq_axis_kernel_size=hparams.freq_axis_kernel_size,
    scalar_input=True,
    legacy=hparams.legacy,
    )
    return model

def wavegen(model, c=None, tqdm=tqdm):
    """
    Generate waveform samples by WaveNet.
    """

    model.eval()
    model.make_generation_fast_()

    Tc = c.shape[0]
    upsample_factor = hparams.hop_size
    # Overwrite length according to feature size
    length = Tc * upsample_factor

    # B x C x T
    c = torch.FloatTensor(c.T).unsqueeze(0)

    initial_input = torch.zeros(1, 1, 1).fill_(0.0)

    # Transform data to GPU
    initial_input = initial_input.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat


# load pretrained model
c = "20180510_mixture_lj_checkpoint_step000320000_ema.pth"
checkpoint = torch.load(c, map_location=device)
model = build_model().to(device)
model.load_state_dict(checkpoint["state_dict"])

# output_audio has fs = 22 kHz
spect = wav_to_mel_spectrogram('output_audio.wav', hp) # This makes it a spectrogram
librosa.display.specshow(spect, sr=hparams.sample_rate, x_axis='time',
                         y_axis='hz', hop_length=hparams.hop_size, cmap='magma')
c = spect[:128, :]
print(c.shape)

waveform = wavegen(model, c=c) # This takes the spectrogram
sf.write('reconstructed_audio.wav', waveform, 16000, 'PCM_24')