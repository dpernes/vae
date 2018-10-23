import torch
from torch import nn
from torch import optim
from torchvision import transforms
import numpy as np
import sys
import argparse
from vae import VAE, vae_loss
from imgdataset import ImgDataset
from utils import imsave


parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--data-path', type=str,
                    default='/data/DB/celebA/img_align_celeba/',
                    help='path for the images dir')
parser.add_argument('--img-crop', type=int, default=148,
                    help='size for center cropping (default: 148)')
parser.add_argument('--img-resize', type=int, default=64,
                    help='size for resizing (default: 64)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--valid_split', type=float, default=.2,
                    help='fraction of data for validation (default: 0.2)')
parser.add_argument('--kl-weight', type=float, default=1e-3,
                    help='weight of the KL loss (default: 1e-3)')
parser.add_argument('--filters', type=str, default='64, 128, 256, 512',
                    help=('number of filters for each conv. layer (default: '
                          + '\'64, 128, 256, 512\')'))
parser.add_argument('--kernel_sizes', type=str, default='3, 3, 3, 3',
                    help=('kernel sizes for each conv. layer (default: '
                          + '\'3, 3, 3, 3\')'))
parser.add_argument('--strides', type=str, default='2, 2, 2, 2',
                    help=('strides for each conv. layer (default: \'2, 2, 2, '
                          + '2\')'))
parser.add_argument('--latent-dim', type=int, default=128,
                    help='latent space dimension (default: 128)')
parser.add_argument('--batch-norm', type=int, default=1,
                    help=('whether to use or not batch normalization (default:'
                          + ' 1)'))
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
args = parser.parse_args()
args.filters = [int(item) for item in args.filters.split(',')]
args.kernel_sizes = [int(item) for item in args.kernel_sizes.split(',')]
args.strides = [int(item) for item in args.strides.split(',')]
args.batch_norm = bool(args.batch_norm)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def train(vae, optimizer, train_loader, n_epochs, kl_weight=1e-3,
          valid_loader=None, n_gen=0):

    device = next(vae.parameters()).device
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))

        # training phase
        vae.train()  # training mode
        for i, X in enumerate(train_loader):
            X = X.to(device)

            # forward pass
            Xrec, z_mean, z_logvar = vae(X)

            # loss, backward pass and optimization step
            loss, reconst_loss, kl_loss = vae_loss(Xrec, X, z_mean, z_logvar,
                                                   kl_weight=kl_weight)
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            sys.stdout.write(
              '\r'
              + '........{} mini-batch loss: {:.3f} |'
                .format(i + 1, loss.item())
              + ' reconst loss: {:.3f} |'
                .format(reconst_loss.item())
              + ' kl loss: {:.3f}'
                .format(kl_loss.item()))
            sys.stdout.flush()

        torch.save(vae.state_dict(), './models/vae.pth')

        # evaluation phase
        print()
        with torch.no_grad():
            vae.eval()  # inference mode

            # compute training loss
            train_loss = 0.
            for i, X in enumerate(train_loader):
                X = X.to(device)

                Xrec, z_mean, z_logvar = vae(X)
                train_loss += vae_loss(Xrec, X, z_mean, z_logvar,
                                       kl_weight=kl_weight)[0]

                # save original and reconstructed images
                if i == 0:
                    imsave(X, './imgs/train_orig.png')
                    imsave(Xrec, './imgs/train_rec.png')

            train_loss /= i + 1
            print('....train loss = {:.3f}'.format(train_loss.item()))

            if valid_loader is None:
                print()
            else:  # compute validation loss
                valid_loss = 0.
                for i, X in enumerate(valid_loader):
                    X = X.to(device)

                    Xrec, z_mean, z_logvar = vae(X)
                    valid_loss += vae_loss(Xrec, X, z_mean, z_logvar,
                                           kl_weight=kl_weight)[0]

                    # save original and reconstructed images
                    if i == 0:
                        imsave(X, './imgs/valid_orig.png')
                        imsave(Xrec, './imgs/valid_rec.png')

                valid_loss /= i + 1
                print('....valid loss = {:.3f}'.format(valid_loss.item()))
                print()

            # generate some new examples
            if n_gen > 0:
                z = torch.randn((n_gen, vae.latent_dim)).to(device)
                Xnew = vae.decoder(z)
                imsave(Xnew, './imgs/gen.png')


SetRange = transforms.Lambda(lambda X: 2*X - 1.)
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop((args.img_crop,
                                                       args.img_crop)),
                                transforms.Resize((args.img_resize,
                                                   args.img_resize)),
                                transforms.ToTensor(),
                                SetRange])
dataset = ImgDataset(args.data_path, transform=transform)

# creating data indices for training and validation splits
dataset_size = len(dataset)  # number of samples in training + validation sets
indices = list(range(dataset_size))
split = int(np.floor(args.valid_split * dataset_size))  # samples in valid. set
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=4,
                                           sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=4,
                                           sampler=valid_sampler)

print('{} samples for training'
      .format(int((1 - args.valid_split) * dataset_size)))
print('{} samples for validation'
      .format(int(args.valid_split * dataset_size)))

img_channels = dataset[0].shape[0]

vae = VAE(img_channels,
          args.img_resize,
          args.latent_dim,
          args.filters,
          args.kernel_sizes,
          args.strides,
          activation=nn.LeakyReLU,
          out_activation=nn.Tanh,
          batch_norm=args.batch_norm).to(DEVICE)
print(vae)

optimizer = optim.Adam(vae.parameters(),
                       lr=args.lr,
                       weight_decay=.0)

train(vae, optimizer, train_loader, args.epochs, kl_weight=args.kl_weight,
      valid_loader=valid_loader, n_gen=args.batch_size)
