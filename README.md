# Variational Autoencoder
Variational Autoencoder (VAE) implemented in Pytorch.

## Model architecture
The architecture of the VAE is customisable via command line, run ``train_vae.py --help`` for more details.

**constraint:** the architecture of the decoder is the transposed of the encoder's.

## Results for CelebA dataset
The VAE with the default parameters was trained on CelebA dataset. This pre-trained model is available in the ``models``directory in this repository.
