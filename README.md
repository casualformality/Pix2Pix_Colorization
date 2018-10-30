# Modified Pix2Pix neural network tailored for image colorization

Code adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py

Original Pix2Pix implementation (in Lua): https://github.com/phillipi/pix2pix

The included networks use an architecture with 2x2 average pooling after every Conv2D layer, and Upsampling x4 instead of x2 on each deconv layer. Compare your results to the original Pix2Pix implementation!