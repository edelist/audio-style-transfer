"""
Audio Style Transfer 
- neural style transfer with VGG backbone for combining content and style spectrograms 
- WaveNet for audio generation conditioned on spectrogram data
"""

from __future__ import print_function

import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
import soundfile as sf

from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms

from utils import GetDictWithDotNotation, imshow, pad_style_img, mel_spect_to_image
from audio_utils import wav_to_mel_spectrogram
from style_transfer import run_style_transfer
from wavegen import build_model, wavegen

# Suppress all warnings
warnings.filterwarnings("ignore")

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
        # "sampling_rate": 22050,
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
            # 'sample_rate': 22050,
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
            'gate_channels': 512,  # split into 2 groups internally for gated activation
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

hparams = hp.m_wave_net.hp

# load pretrained model
c = "20180510_mixture_lj_checkpoint_step000320000_ema.pth"
checkpoint = torch.load(c, map_location=device)
model = build_model(hparams).to(device)
model.load_state_dict(checkpoint["state_dict"])

# Create mel spectrograms from content, style audio
DURATION = 5
CONTENT_PATH = 'test_data/content1.wav'
STYLE_PATH = 'test_data/style2.wav'
spect_content = wav_to_mel_spectrogram(CONTENT_PATH, hp, DURATION)
spect_style = wav_to_mel_spectrogram(STYLE_PATH, hp, DURATION)

if spect_style.shape[0] < spect_content.shape[0]:
    spect_style = pad_style_img(spect_content, spect_style)
else:
    spect_content = pad_style_img(spect_style, spect_content)

CONTENT_STR = 'content1.png'
STYLE_STR = 'style1.png'
mel_spect_to_image(spect_content, save=True, save_str=CONTENT_STR)
mel_spect_to_image(spect_style, save=True, save_str=STYLE_STR)

loader = transforms.Compose([
    # transforms.Resize(imsize),   # scale imported image
    transforms.ToTensor()])        # transform it into a torch tensor

def image_loader(image_name):
    """Load image in dimension for style transfer"""
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Load images for style transfer
style_img = image_loader(STYLE_STR).expand(-1,3,-1,-1)
content_img = image_loader(CONTENT_STR).expand(-1,3,-1,-1)
# style_img = image_loader('Results/pngs/style2_out.png')
# content_img = image_loader('Results/pngs/content1_out.png')

# Define cnn model
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
cnn.to(device)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_selected = ['conv_4']
style_layers_selected = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
STYLE_WEIGHT=1000000
input_img = content_img.clone().detach().requires_grad_(True)

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, content_layers=content_layers_selected, 
                            style_layers=style_layers_selected, device=device, num_steps=100, style_weight=STYLE_WEIGHT)


# Remove dummy batch dimension, convert to grayscale
output = torch.mean(output.squeeze(0), dim=0)
output_img = output.detach().numpy()
output_img *= 255
output_img = Image.fromarray(output_img.astype(np.uint8))
output_img = output_img.convert('L')
output_img.save('output_content1_style1.png')

# Show new spectrogram
# plt.figure()
# imshow(output, title='Output Image')
# plt.show()

output_sr = Image.open('output_content1_style1.png')
width, height = output_sr.size
aspect_ratio = width/height

target_width = 80
# Calculate the new width based on the target height
target_height = int(aspect_ratio * target_width)

# Resize the image
output_sr = output_sr.resize((target_height, target_width))
transform = transforms.ToTensor()
output_sr = transform(output_sr).squeeze(0) # 2065, 280

# waveform = wavegen(model, hparams, device, c=output_sr[:128, :])
waveform = wavegen(model, hparams, device, c=output)
output_str = 'Results/wavs/output_sr_content1_style2_v2.wav'
sf.write(output_str, waveform, 16000, 'PCM_24')
