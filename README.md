# Audio-Style-Transfer

  The task is to perform audio style transfer through the use of Mel spectrograms with a CNN using a VGG-19. The first milestone, this status report, uses a VGG visual pre-trained style transfer model, and writes scripts that prepare the training data (audio-to-spectrogram) and perform the final conversion into an audible output format (spectrogram-to-audio) through the use of a pre-trained WaveNet vocoder.

  In order to validate our approach, we have started by performing style transfer using the algorithm proposed by Gatys et al in [6], which applies the artistic “style” of The desired style and content of the reference images are captured by hidden layers of the VGG-19 image classification network (a pre-trained version of this model is available in the torchvision.models library). Beginning with an image of pure noise, this method iteratively optimizes a loss value composed of a weighted sum of content and style losses. The content loss is simply the mean squared error between the representations of the original and generated images at predetermined layers in the network. To represent the style at a given layer of the network, we compute the Gram matrix of that layer , which is the inner product between the feature vectors of the layer. Then, for each layer, the style loss is computed as the mean squared difference between the style representations of the generated (a, Al) and original (x, Gl) images.

## Install dependencies
```bash
pip install torch torchvision pillow librosa soundfile webrtcvad tqdm wavenet_vocoder 
```

### Usage 
Download Wavenet pretrained weights [here](https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth?e=1&dl=0)


1. Single Layer CNN (style transfer) with Griffin Lim algorithm (audio reconstruction):

2. Neural style transfer with VGG backbone + WaveNet (audio reconstruction):

```bash
$ python vgg-wavenet.py
```

3. Neural style transfer with ViT backbone + WaveNet (audio reconstruction):

```bash
$ jupyter Vision_Transformer.ipynb
```

4. Neural style transfer with ResNet backbone + WaveNet (audio reconstruction):

```bash
$ jupyter EC523_FinalProject_Script_ResNet.ipynb
```