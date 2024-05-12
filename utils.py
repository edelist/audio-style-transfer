"""
Utils for VGG - WaveNet audio style transfer
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def gram_matrix(input):
    """Compute gram matrix"""
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # inner product of
    return G.div(a * b * c * d)           # divide by layer dimension

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    resized_image = original_image.resize(size)
    resized_image.save(output_image_path)

def imshow(tensor, title=None):
    """show image"""
    array = tensor.detach().numpy()
    plt.imshow(array, cmap='gray')
    if title is not None:
        plt.title(title)

def imshow_color(tensor, title=None):
    """Show images"""
    unloader = transforms.ToPILImage()  # reconvert into PIL image  

    image = tensor.cpu().clone()    # we clone the tensor to not do changes on it
    image = image.squeeze(0)        # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

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

def mel_spect_to_image(spect, show=False, title=None, save=False, save_str=None):
    """convert mel spectrogram to Pillow image"""
    spect_gray = np.dot(spect,255)
    image = Image.fromarray(spect_gray)
    gray_image = image.convert("L")

    # Show image
    if show:
        plt.imshow(gray_image, cmap='gray')
        plt.show()
        if title:
            plt.title(title)

    # Save image
    if save:
        gray_image.save(save_str, quality=95)

    return gray_image

def pad_style_img(content_img, style_img):
    """ 
    Copies the style image to match the shape of the content image;
    content and style images are numpy arrays
    """
    content_row = content_img.shape[0]
    style_row = style_img.shape[0]
    pad_num = (content_row - style_row) // style_row
    pad_rem = (content_row - style_row) - pad_num*style_row

    img_style_pad = style_img
    for i in range(pad_num):
        img_style_pad = np.vstack((img_style_pad, style_img))
    img_style_rem = style_img[:pad_rem, :]
    img_style_pad = np.vstack((img_style_pad, img_style_rem))

    return img_style_pad

def show_color_image(tensor):
    # Convert the tensor to a NumPy array
    numpy_array = tensor.numpy()

    # Transpose the array to match Matplotlib's expected format (H x W x C)
    numpy_array = numpy_array.transpose(1, 2, 0)

    # Display the color image using Matplotlib
    plt.imshow(numpy_array[0,:,:])

# def rgb_to_grayscale(rgb):
#     """Convert RGB image to grayscale by taking average of 3 channels"""
#     image = rgb.clone()
#     output = torch.mean(image, axis=0)

#     return torch.mean(image,ax)
    
# tensor_3d = torch.rand(5, 4, 3)
# output = rgb_to_grayscale(tensor_3d)
# print(output.size())
