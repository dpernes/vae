import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, in_dim, latent_dim, filters,
                 kernel_sizes, strides, paddings, flat_dim,
                 activation=nn.LeakyReLU, batch_norm=True):
        '''
        in_channels (int): number of channels of the input image (e.g.: 1 for
                           grayscale, 3 for color images).
        in_dim (int): number of pixels on each row / column of the input image
                      (assumes the images are square).
        latent_dim (int): dimension of the output (latent) space.
        flat_dim (int): flattened dimension after the last conv. layer.
        filters (list of length n_conv): number of filters for each conv.
                                         layer.
        kernel_sizes (list of length n_conv): kernel size for each conv. layer.
        strides (list of length n_conv): strides for each conv. layer.
        paddings (list of length n_conv): zero padding added to the input for
                                          each conv. layer.
        activation (subclass of nn.Module): activation used in all layers,
                                            except in the output (default:
                                            LeakyReLU).
        batch_norm (boolean): if True, batch normalization is applied in every
                              layer before the activation (default: True).
        '''
        super(Encoder, self).__init__()

        self.in_dim = in_dim
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.flat_dim = flat_dim
        self.activation = activation
        self.batch_norm = batch_norm

        n_conv = len(self.filters)

        # first conv. layer
        conv_layers = nn.ModuleList([nn.Conv2d(self.in_channels,
                                    self.filters[0],
                                    self.kernel_sizes[0],
                                    stride=self.strides[0],
                                    padding=self.paddings[0])])
        if self.batch_norm:
            conv_layers.append(nn.BatchNorm2d(self.filters[0]))
        conv_layers.append(self.activation())

        # remaining conv. layers
        for i in range(1, n_conv):
            layer = nn.ModuleList([nn.Conv2d(self.filters[i-1],
                                             self.filters[i],
                                             self.kernel_sizes[i],
                                             stride=self.strides[i],
                                             padding=self.paddings[i])])
            if self.batch_norm:
                layer.append(nn.BatchNorm2d(self.filters[i]))
            layer.append(self.activation())
            conv_layers.extend(layer)

        # connect all conv. layers in a sequential block
        self.conv_block = nn.Sequential(*conv_layers)

        # define mean and variance layers
        self.mean_block = nn.Linear(self.flat_dim, self.latent_dim)
        self.logvar_block = nn.Linear(self.flat_dim, self.latent_dim)

        self.param_init()

    def param_init(self):
        for layer in self.modules():
            if hasattr(layer, 'weight'):
                if type(layer) == nn.BatchNorm2d:
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)

    def forward(self, X):
        h = self.conv_block(X)
        h = h.reshape(-1, self.flat_dim)
        z_mean = self.mean_block(h)
        z_logvar = self.logvar_block(h)

        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, in_channels, in_dim, filters,
                 kernel_sizes, strides, paddings, out_paddings,
                 activation=nn.LeakyReLU, out_activation=nn.Tanh,
                 batch_norm=True):
        '''
        latent_dim (int): dimension of the input (latent) space.
        in_channels (int): number of channels of the input of the first
                           transposed convolution.
        in_dim (int): number of pixels on each row / column of the input of the
                      first transposed convolution (assumes the images are
                      square).
        filters (list of length n_conv): number of filters for each transp.
                                         conv. layer.
        kernel_sizes (list of length n_conv): kernel size for each transp.
                                              conv. layer.
        strides (list of length n_conv): strides for each transp. conv. layer.
        paddings (list of length n_conv): zero padding added to the input for
                                          each transp. conv. layer.
        out_paddings (list of length n_conv): output padding for each transp.
                                              conv. layer.
        activation (subclass of nn.Module): activation used in all layers,
                                            except in the output (default:
                                            LeakyReLU).
        out_activation (subclass of nn.Module): activation used in the output
                                                layer (default: Tanh).
        batch_norm (boolean): if True, batch normalization is applied (default:
                              True).
        '''
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.out_paddings = out_paddings
        self.activation = activation
        self.out_activation = out_activation
        self.batch_norm = batch_norm

        n_conv = len(self.filters)

        # input layer
        flat_dim = self.in_channels * (self.in_dim**2)
        input_layers = nn.ModuleList([nn.Linear(self.latent_dim, flat_dim)])
        if self.batch_norm:
            input_layers.append(nn.BatchNorm1d(flat_dim))
        input_layers.append(self.activation())

        self.input_block = nn.Sequential(*input_layers)

        # upsampling layers
        upsample_layers = nn.ModuleList([nn.ConvTranspose2d(
                                           self.in_channels,
                                           self.filters[0],
                                           self.kernel_sizes[0],
                                           stride=self.strides[0],
                                           padding=self.paddings[0],
                                           output_padding=out_paddings[0])])
        if self.batch_norm:
                upsample_layers.append(nn.BatchNorm2d(self.filters[0]))
        upsample_layers.append(self.activation())
        for i in range(1, n_conv):
            upsample_layers.append(nn.ConvTranspose2d(
                                     self.filters[i-1],
                                     self.filters[i],
                                     self.kernel_sizes[i],
                                     stride=self.strides[i],
                                     padding=self.paddings[i],
                                     output_padding=out_paddings[i]))

            if i < n_conv-1:
                if self.batch_norm:
                    upsample_layers.append(nn.BatchNorm2d(self.filters[i]))
                upsample_layers.append(self.activation())
            else:
                upsample_layers.append(self.out_activation())

        # connect all upsampling layers in a sequential block
        self.upsample_block = nn.Sequential(*upsample_layers)

        self.param_init()

    def param_init(self):
        for layer in self.modules():
            if hasattr(layer, 'weight'):
                if type(layer) in (nn.BatchNorm1d, nn.BatchNorm2d):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)

    def forward(self, z):
        h = self.input_block(z)
        h = h.reshape(-1, self.in_channels, self.in_dim, self.in_dim)
        Xrec = self.upsample_block(h)

        return Xrec


class VAE(nn.Module):
    def __init__(self, img_channels, img_dim, latent_dim, filters,
                 kernel_sizes, strides, activation=nn.LeakyReLU,
                 out_activation=nn.Tanh, batch_norm=True):
        '''
        in_dim (int): number of pixels on each row / column of the images
                      (assumes the images are square).
        in_channels (int): number of channels of the images (e.g.: 1 for
                           grayscale, 3 for color images).
        latent_dim (int): dimension of the latent space.
        filters (list of length n_conv): number of filters for each conv.
                                         layer.
        kernel_sizes (list of length n_conv): kernel size for each conv.
        strides (list of length n_conv): strides for each conv. layer.
        activation (nn.Module): activation used in all layers (default:
                                LeakyReLU).
        out_activation (subclass of nn.Module): activation used in the output
                                                layer (default: Tanh).
        batch_norm (boolean): if True, batch normalization is applied in every
                              layer before the activation (default: True).
        '''
        super(VAE, self).__init__()

        self.img_dim = img_dim
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.activation = activation
        self.out_activation = out_activation
        self.batch_norm = batch_norm

        n_conv = len(self.filters)

        # compute the paddings and the flattened dimension at the output of the
        # last conv.
        paddings = []
        dims = [self.img_dim]
        for i in range(n_conv):
            if (dims[i] - self.kernel_sizes[i]) % strides[i] == 0:
                paddings.append((self.kernel_sizes[i] - 1)//2)
            else:
                paddings.append((self.kernel_sizes[i] - strides[i] + 1)//2)

            dims.append((dims[i] + 2*paddings[i] - self.kernel_sizes[i])
                        // self.strides[i] + 1)
        flat_dim = self.filters[-1] * (dims[-1]**2)

        self.encoder = Encoder(self.img_channels, self.img_dim,
                               self.latent_dim,  self.filters,
                               self.kernel_sizes, self.strides,
                               paddings, flat_dim,
                               activation=self.activation,
                               batch_norm=self.batch_norm)

        # the decoder architecture will be the transposed of the encoder's
        filters_dec = (list(reversed(self.filters[0:n_conv-1]))
                         + [img_channels])
        kernel_sizes_dec = list(reversed(self.kernel_sizes))
        strides_dec = list(reversed(self.strides))
        paddings = list(reversed(paddings))
        dims = list(reversed(dims))

        # compute the output paddings
        out_paddings = []
        for i in range(n_conv):
            out_dim = ((dims[i] - 1)*strides_dec[i] - 2*paddings[i] +
                       kernel_sizes_dec[i])
            out_paddings.append(dims[i+1] - out_dim)

        self.decoder = Decoder(self.latent_dim, self.filters[-1], dims[0],
                               filters_dec, kernel_sizes_dec, strides_dec,
                               paddings=paddings, out_paddings=out_paddings,
                               activation=self.activation,
                               out_activation=self.out_activation,
                               batch_norm=self.batch_norm)

    def sample(self, z_mean, z_logvar):
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(.5*z_logvar) * eps

        return z

    def forward(self, X):
        z_mean, z_logvar = self.encoder(X)
        z = self.sample(z_mean, z_logvar)
        Xrec = self.decoder(z)

        return Xrec, z_mean, z_logvar


def vae_loss(Xrec, X, z_mean, z_logvar, kl_weight=1e-3):
    reconst_ls = F.mse_loss(Xrec, X)
    kl_ls = torch.mean(-.5*torch.sum(1 + z_logvar - z_mean**2
                                     - torch.exp(z_logvar), dim=1), dim=0)

    loss = reconst_ls + kl_weight * kl_ls

    return loss, reconst_ls, kl_ls
