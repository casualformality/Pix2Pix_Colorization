#!/usr/bin/env python
import numpy as np
import os
import datetime
from glob import glob
from array import array
from imageio import imread, imsave
import struct

from keras.layers import Dropout, Concatenate, Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Model


def conv_2d_layer(input_layer, filters, f_size=4, normalize=True):
    # Stride of 2 downsamples our inputs while still learning useful parameters
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(input_layer)
    # Leaky ReLU here give good non-linear response while maintaining gradients if < 0
    d = LeakyReLU(alpha=0.2)(d)
    # Normalization is not useful for the first layer, so this is taken as an argument
    if normalize:
        d = BatchNormalization(momentum=0.8)(d)
    return d


def deconv_2d_layer(input_layer, input_skip, filters, f_size=4, dropout=0.0):
    # Upsample the input by a factor of 2 ot compensate for downsampling in the Conv2D layers
    u = UpSampling2D(size=2)(input_layer)
    # Standard ReLU is fine here, since these layers are more about reconstruction than learning
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    # Include option for dropout to improve network generalizeability
    if dropout:
        u = Dropout(dropout)(u)
    # Batch normalization is default here, since deconv layers won't be at the input layer
    u = BatchNormalization(momentum=0.8)(u)
    # Use skip layers to maintain structure of original image
    u = Concatenate()([u, input_skip])
    return u


def build_generator(img_shape, _filter_depth, _num_layers, num_channels=3):
    """
    This generator is for image-to-image translation, and uses a series
    of downsampling and upsampling convolutional layers to achieve this.

    Skip layers are used to push high-level structure through the network,
    so the generator doesn't have to do much to maintain the objects in the
    input images.
    """
    input_layer = Input(shape=img_shape)
    new_layer = input_layer

    # Construct downsampling layers
    down_layers = []
    should_normalize = False
    for ll in range(_num_layers):
        # Add depth to filters as we downsample
        filt_size = min(2 ** ll, 8) * _filter_depth
        # Add new downsampling filter
        new_layer = conv_2d_layer(new_layer, filt_size, normalize=should_normalize)
        # Loop maintenance
        down_layers.append(new_layer)
        should_normalize = True

    # Construct matching upsampling layers
    for ll in range(_num_layers - 1):
        # Remove depth to filters as we upsample
        filt_size = min(2 ** (_num_layers - 2 - ll), 8) * _filter_depth
        new_layer = deconv_2d_layer(new_layer, down_layers[-2 - ll], filt_size, dropout=0.2)
    # Drop the normalization and skip input on the last step
    new_layer = UpSampling2D(size=2)(new_layer)
    # Stride 1 and same padding keep the output the same size
    # This is a regression problem, but we want to limit the output range, so sigmoid makes the most sense here
    output_layer = Conv2D(num_channels, kernel_size=4, strides=1, padding='same', activation='sigmoid')(new_layer)

    return Model(input_layer, output_layer)


def build_discriminator(_cond_img_shape, _targ_img_shape, _filter_depth, _num_layers):
    """
    This network is responsible for discriminating between real and generated
    inputs. This is achieved by concatenating the source and conditional
    images

    Since the discriminator is only responsible for binary classification,
    we don't need any upsampling layers, and the network can be a bit simpler
    in general.
    """
    targ_input = Input(shape=_targ_img_shape)  # Source image
    cond_input = Input(shape=_cond_img_shape)  # Conditional image

    input_layer = Concatenate(axis=-1)([targ_input, cond_input])
    new_layer = input_layer

    for ll in range(_num_layers):
        # Add depth to filters as we downsample
        filt_size = min(2 ** ll, 8) * _filter_depth
        new_layer = conv_2d_layer(new_layer, filt_size)

    # Output tensor consists of a single layer with a patch for the input and
    # conditional image sources
    output_layer = Conv2D(1, kernel_size=4, strides=1, padding='same')(new_layer)

    return Model([targ_input, cond_input], output_layer)


def create_networks(_cond_img_shape, _targ_img_shape, _filter_depth, _num_layers, _num_channels):
    """
    Adam is a good optimizer for both networks, since it combines the benefits
    of RMS and Momentum optimization techniques.
    """
    optimizer = Adam(0.0002, 0.5)

    """
    This is effectively a binary classification problem with images patches
    as output. MSE loss reduces the difference between predicted and true
    pixel-wise classification values, and the accuracy metric maximizes
    the prediction accuracy of the system.
    """
    _discriminator = build_discriminator(_cond_img_shape, _targ_img_shape, filter_depth, _num_layers)
    _discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    targ_input = Input(shape=_targ_img_shape)
    cond_input = Input(shape=_cond_img_shape)

    """
    Configure the generator to modify the conditional images to create
    artificial versions of the original.
    """
    _generator = build_generator(_cond_img_shape, _filter_depth, _num_layers, _num_channels)
    fake_input = _generator(cond_input)

    """
    Compile a new model that feeds the trainable generator network into a
    frozen version of the discriminator network. This allows us to optimize
    the generator on each batch iteration to produce more realistic outputs.

    Here, MSE*1 is used to minimize the output of the discriminator (in order
    to fool it) and MAE*100 is used to minimize the difference between the
    target and generated images.
    """
    _discriminator.trainable = False
    fake_output = _discriminator([fake_input, cond_input])
    _combined = Model(inputs=[targ_input, cond_input], outputs=[fake_output, fake_input])
    _combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)

    return _combined, _generator, _discriminator


def get_image_files(directory, _batch_size):
    _cond_files = glob('/'.join([directory, 'bw/*']))
    _targ_files = glob('/'.join([directory, 'color/*']))

    num_samples = len(_cond_files)
    num_samples = num_samples - (num_samples % _batch_size)
    num_batches = int(np.floor(num_samples / _batch_size))

    return _cond_files, _targ_files, num_samples, num_batches


def load_image_batch_rb(_cond_files, _targ_files, _indexes, _image_size):
    _batch_size = len(_indexes)

    cond_batch = np.zeros((_batch_size, _image_size, _image_size, 1))  # BW photos have 1 channel
    targ_batch = np.zeros((_batch_size, _image_size, _image_size, 3))  # Data for Red and Blue channels

    for idx in range(len(_indexes)):
        img_idx = _indexes[idx]
        cond_batch[idx, :, :, 0] = imread(_cond_files[img_idx]) / 255.0
        img = imread(_targ_files[img_idx])
        targ_batch[idx, :, :, :] = img / 255.0  # Ignore green channel

    return cond_batch, targ_batch


def train_networks(_combined, _generator, _discriminator, _directory,
                   _image_size, _num_channels, _num_epochs, _batch_size):

    # [cond_imgs, targ_imgs, num_samples, num_batches] = get_images_mnist('./train-images.idx3-ubyte', _batch_size)
    [cond_files, targ_files, num_samples, num_batches] = get_image_files(_directory, _batch_size)
    batch_idxs = np.random.choice(range(num_samples), size=[num_batches, _batch_size])

    # Labels for image patches output by discriminator
    patch_size = int(_image_size / 2 ** num_layers)
    real_labels = np.ones((_batch_size,) + (patch_size, patch_size, 1))
    fake_labels = np.zeros((_batch_size,) + (patch_size, patch_size, 1))

    start_time = datetime.datetime.now()

    for epoch in range(_num_epochs):
        for batch in range(num_batches):
            # cond_batch, targ_batch = load_image_batch_mnist(cond_imgs, targ_imgs, batch_idxs[batch])
            cond_batch, targ_batch = load_image_batch_rb(cond_files, targ_files, batch_idxs[batch], _image_size)

            """
            Discriminator Update Step
            """
            # Generate transformed images based on conditional input
            fake_batch = _generator.predict(cond_batch)

            # Average the discriminator loss across real and generated inputs
            disc_loss_real = _discriminator.train_on_batch([targ_batch[:, :, :, [0, 2]], cond_batch], real_labels)
            disc_loss_fake = _discriminator.train_on_batch([fake_batch, cond_batch], fake_labels)
            disc_loss = 0.5 * np.add(disc_loss_fake, disc_loss_real)

            """
            Generator Update Step
            """
            # Train the combined network with ideal discriminator labels and
            # ideal transformed images as ideal outputs
            gen_loss = _combined.train_on_batch([targ_batch[:, :, :, [0, 2]], cond_batch],
                                                [real_labels, targ_batch[:, :, :, [0, 2]]])

            time_elapsed = datetime.datetime.now() - start_time

            # Print the progress
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, _num_epochs,
                                                                                                  batch, num_batches,
                                                                                                  disc_loss[0],
                                                                                                  100 * disc_loss[1],
                                                                                                  gen_loss[0],
                                                                                                  time_elapsed))

            # If at save interval => save generated image samples
            if batch % 2000 == 0:
                generate_samples(generator, epoch, cond_batch[0:8], targ_batch[0:8])


def post_process_samples_rb(cond_samples, fake_samples, targ_samples):
    # Y = 0.2126R + 0.7152G + 0.0722B
    # Apply inverse HSV to find blue channel on fake and target samples
    fake_samples_g = (cond_samples[:, :, :, 0] -
                      0.2126*fake_samples[:, :, :, 0] -
                      0.0722*fake_samples[:, :, :, 1]) / 0.7152
    np.maximum(fake_samples_g, 0, fake_samples_g)
    fake_samples = np.insert(fake_samples, 1, fake_samples_g, axis=3)

    # Extend cond_samples to create "grayscale" RGB image
    cond_samples = cond_samples.repeat(3, axis=3)

    # Generate composite image of three samples
    composite_images = np.concatenate([cond_samples, fake_samples, targ_samples], axis=2)
    np.multiply(composite_images, 255.0, out=composite_images)
    # Convert from [0.0-1.0] to UINT8 [0-255]
    composite_images = composite_images.astype('uint8', copy=False)

    return composite_images


def generate_samples(_generator, prefix, cond_samples, targ_samples):
    os.makedirs('outputs', exist_ok=True)
    fake_samples = _generator.predict(cond_samples)

    composite_images = post_process_samples_rb(cond_samples, fake_samples, targ_samples)

    for idx in range(len(composite_images)):
        imsave("outputs/%d_%d.png" % (prefix, idx), composite_images[idx])


if __name__ == "__main__":
    """
    These parameters are temporary values for use with the MNIST digit database

    This test will attempt to train the generator to invert the colors on the digits
    """
    """ MNIST Training
    img_shape = (28, 28, 1)
    filter_depth = 12
    num_layers = 2
    """
    cond_img_shape = (128, 128, 1)
    targ_img_shape = (128, 128, 2)
    filter_depth = 64
    num_layers = 6
    num_epochs = 200
    batch_size = 10

    # Create and compile the networks
    [combined, generator, discriminator] = create_networks(cond_img_shape,
                                                           targ_img_shape,
                                                           filter_depth,
                                                           num_layers,
                                                           targ_img_shape[2])

    print(combined.summary())

    # Train and save the networks
    train_networks(combined, generator, discriminator, './bird_med',
                   targ_img_shape[0], targ_img_shape[2], num_epochs, batch_size)

    combined_json = combined.to_json()
    with open("combined.json", "w") as json_file:
        json_file.write(combined_json)
    combined.save_weights("combined.h5")

    discriminator_json = discriminator.to_json()
    with open("discriminator.json", "w") as json_file:
        json_file.write(discriminator_json)
    discriminator.save_weights("discriminator.h5")

    generator_json = generator.to_json()
    with open("generator.json", "w") as json_file:
        json_file.write(generator_json)
    generator.save_weights("generator.h5")

