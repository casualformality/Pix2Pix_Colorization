#!/usr/bin/env python
import numpy as np
import os
from glob import glob
from imageio import imread, imsave

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import model_from_json


def load_image_batch_rb(_cond_files, _targ_files, _indexes, _image_size):
    _batch_size = len(_indexes)

    _cond_batch = np.zeros((_batch_size, _image_size, _image_size, 1))  # BW photos have 1 channel
    _targ_batch = np.zeros((_batch_size, _image_size, _image_size, 3))  # Data for Red and Blue channels

    for idx in range(len(_indexes)):
        img_idx = _indexes[idx]
        _cond_batch[idx, :, :, 0] = imread(_cond_files[img_idx]) / 255.0
        img = imread(_targ_files[img_idx])
        _targ_batch[idx, :, :, :] = img / 255.0  # Ignore green channel

    return _cond_batch, _targ_batch


def get_image_files(_directory, _batch_size):
    _cond_files = glob('/'.join([_directory, 'bw/*']))
    _targ_files = glob('/'.join([_directory, 'color/*']))

    _num_samples = len(_cond_files)
    _num_samples = _num_samples - (_num_samples % _batch_size)
    _num_batches = int(np.floor(_num_samples / _batch_size))

    return _cond_files, _targ_files, _num_samples, _num_batches


def load_model(_directory=".", _suffix=""):
    with open("%s/generator%s.json" % (_directory, _suffix), "r") as _json_file:
        _generator_json = _json_file.read()
    _generator = model_from_json(_generator_json)
    _generator.load_weights("%s/generator%s.h5" % (_directory, _suffix))

    return _generator


def post_process_samples_rb(cond_samples, fake_samples, targ_samples):
    # Y= 0.2126R+ 0.7152G+ 0.0722B
    # Apply inverse HSV to find blue channel on fake and target samples
    fake_samples_g = (cond_samples[:, :, :, 0] -
                      0.2126*fake_samples[:, :, :, 0] -
                      0.0722*fake_samples[:, :, :, 1]) / 0.7152
    np.maximum(fake_samples_g, 0, fake_samples_g)
    fake_samples = np.insert(fake_samples, 1, fake_samples_g, axis=3)

    # Extend cond_samples to create "grayscale" RGB image
    cond_samples = cond_samples.repeat(3, axis=3)

    composite_images = np.concatenate([cond_samples, fake_samples, targ_samples], axis=2)
    np.multiply(composite_images, 255.0, out=composite_images)
    composite_images = np.maximum(composite_images, 0.0)
    composite_images = np.minimum(composite_images, 255.0)
    # Convert from [0-1] to UINT8 [0-255]
    composite_images = composite_images.astype('uint8', copy=False)

    return composite_images


def generate_samples(_generator, prefix, cond_samples, targ_samples):
    os.makedirs('outputs', exist_ok=True)
    fake_samples = _generator.predict(cond_samples)

    # composite_images = post_process_samples(cond_samples, fake_samples, targ_samples)
    composite_images = post_process_samples_rb(cond_samples, fake_samples, targ_samples)

    for idx in range(len(composite_images)):
        imsave("outputs_gen/%d_%d.png" % (prefix, idx), composite_images[idx])


if __name__ == "__main__":
    batch_size = 32
    directory = "./bird"
    image_size = 256

    generator = load_model()
    [cond_files, targ_files, num_samples, num_batches] = get_image_files(directory, batch_size)
    batch_idxs = np.reshape(range(num_samples), (num_batches, batch_size))

    for batch in range(num_batches):
        cond_batch, targ_batch = load_image_batch_rb(cond_files, targ_files, batch_idxs[batch], image_size)
        fake_batch = generator.predict(cond_batch)
        generate_samples(generator, batch, cond_batch, targ_batch)
