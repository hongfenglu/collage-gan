import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment


def get_early_features(net, noise):
    feat_4 = net.init(noise)
    feat_8 = net.feat_8(feat_4)
    feat_16 = net.feat_16(feat_8)
    feat_32 = net.feat_32(feat_16)
    feat_64 = net.feat_64(feat_32)
    return feat_8, feat_16, feat_32, feat_64

def get_late_features(net, im_size, feat_64, feat_8, feat_16, feat_32):
    feat_128 = net.feat_128(feat_64)
    feat_128 = net.se_128(feat_8, feat_128)

    feat_256 = net.feat_256(feat_128)
    feat_256 = net.se_256(feat_16, feat_256)
    if im_size==256:
        return net.to_big(feat_256)
    
    feat_512 = net.feat_512(feat_256)
    feat_512 = net.se_512(feat_32, feat_512)
    if im_size==512:
        return net.to_big(feat_512)
    
    feat_1024 = net.feat_1024(feat_512)
    return net.to_big(feat_1024)

def get_decoder_mixed(net, im_size, feat_128_a, feat_128_b, feat_8, feat_16, feat_32, feat_64):
    pass

def interpolation(netG, num_steps):
    noise_high = torch.randn(1, nz)
    noise_low = torch.randn(1, nz)
    step_size = (noise_high - noise_low)/(num_steps-1)
    zn = []
    for i in range(num_steps):
        zn.append(noise_low+step_size*i)
    zn = torch.cat(zn).to(device)
    with torch.no_grad():
        g_images, _ = netG(zn)
    print(g_images.shape)
    vutils.save_image(g_images.cpu().add(1).mul(0.5), "./train_results/"+dir_name+"/images/interpolation_1.jpg", nrow=num_steps)
    return 

def latent_traversal(netG, num_steps):
    upper = 2
    infogan_latent_dim = 8
    z = torch.randn(1, 1, infogan_latent_dim)
    z = z.expand(infogan_latent_dim, num_steps+1, -1).clone()
    intervals = [-upper+i*2*upper/num_steps for i in range(num_steps+1)]
    for i in range(infogan_latent_dim):
        for j in range(num_steps+1):
            z[i, j, i] = intervals[j]
    z = z.reshape(-1, infogan_latent_dim)
    fixed_noise_1 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(z.shape[0], 1)
    traversal_z_1 = torch.cat([fixed_noise_1, z], dim=1).to(device)
    fixed_noise_2 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(z.shape[0], 1)
    traversal_z_2 = torch.cat([fixed_noise_2, z], dim=1).to(device)
    fixed_noise_3 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(z.shape[0], 1)
    traversal_z_3 = torch.cat([fixed_noise_3, z], dim=1).to(device)
    with torch.no_grad():
        g_image1, _ = netG(traversal_z_1)
        g_image2, _ = netG(traversal_z_2)
        g_image3, _ = netG(traversal_z_3)
    print(g_image3.shape)
    vutils.save_image(g_image1.cpu().add(1).mul(0.5), "./train_results/"+dir_name+"/images/trav1.jpg", nrow=num_steps+1)
    vutils.save_image(g_image2.cpu().add(1).mul(0.5), "./train_results/"+dir_name+"/images/trav2.jpg", nrow=num_steps+1)
    vutils.save_image(g_image3.cpu().add(1).mul(0.5), "./train_results/"+dir_name+"/images/trav3.jpg", nrow=num_steps+1)
    return

def sample_fixed_latent(netG, num_samples):
    infogan_latent_dim = 8
    noise = torch.FloatTensor(num_samples, nz).normal_(0, 1)
    latent = torch.FloatTensor(1, infogan_latent_dim).uniform_(-1, 1).repeat(num_samples, 1)
    fixed_latent_noise = torch.cat([noise, latent], dim=1).to(device)
    with torch.no_grad():
        g_image1, _ = netG(fixed_latent_noise)

    vutils.save_image(g_image1.cpu().add(1).mul(0.5), "./train_results/"+dir_name+"/images/samples.jpg", nrow=4)

def sample_fixed_noise(netG, num_samples):
    infogan_latent_dim = 8
    latent = torch.FloatTensor(num_samples, infogan_latent_dim).uniform_(-1, 1)
    noise = torch.FloatTensor(1, nz).normal_(0, 1).repeat(num_samples, 1)
    fixed_latent_noise = torch.cat([noise, latent], dim=1).to(device)
    with torch.no_grad():
        g_image1, _ = netG(fixed_latent_noise)

    vutils.save_image(g_image1.cpu().add(1).mul(0.5), "./train_results/"+dir_name+"/images/samples.jpg", nrow=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--dir_name', type=str, default=None)
    parser.add_argument('--ckpt_iter', type=int, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=64)

    args = parser.parse_args()

    ndf = 64
    ngf = 64
    nz = 256 # latent dimension
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    device = 'cuda:0'
    im_size = 256
    latent_dim = 8

    netG = Generator(ngf=ngf, nz=nz, im_size=im_size, infogan_latent_dim=latent_dim)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size, infogan_latent_dim=latent_dim)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    dir_name = args.dir_name
    checkpoint = "./train_results/"+dir_name+"/models/all_%d.pth"%(args.ckpt_iter)
    ckpt = torch.load(checkpoint)
    netG.load_state_dict(ckpt['g'])
    netD.load_state_dict(ckpt['d'])
    avg_param_G = ckpt['g_ema']
    load_params(netG, avg_param_G)
    netG.eval()

    if args.method == 'latent_traversal':
        latent_traversal(netG, args.num_steps-1)
    elif args.method == 'sample_fixed_latent':
        sample_fixed_latent(netG, args.num_samples)
    elif args.method == 'sample_fixed_noise':
        sample_fixed_noise(netG, args.num_samples)
    elif args.method == 'interpolation':
        interpolation(netG, args.num_steps)

