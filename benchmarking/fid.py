import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from torchvision import transforms
from operation import ImageFolder, InfiniteSamplerWrapper
from torch.utils.data import DataLoader

from calc_inception import load_patched_inception_v3
from pytorch_fid import fid_score

@torch.no_grad()
def extract_features(loader, inception, device):
    feature_list = []

    for iteration in tqdm(range(128)):
        img = next(loader)
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=30000)
    parser.add_argument('--path_a', type=str, default='./data/collage/train')
    parser.add_argument('--name', type=str, default='new512')

    args = parser.parse_args()

    inception = load_patched_inception_v3().eval().to(device)
    torch.cuda.empty_cache()

    transform = transforms.Compose(
        [
            transforms.Resize( (args.size, args.size) ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    path_b = './fake/'+args.name+'/eval_%d'%(args.epoch)
    dset_b = ImageFolder(root=path_b, transform=transform)
    loader_b = iter(DataLoader(dset_b, batch_size=args.batch, num_workers=4, sampler=InfiniteSamplerWrapper(dset_b)))


    features_b = extract_features(loader_b, inception, device).numpy()
    print(f'extracted {features_b.shape[1]} features')

    sample_mean = np.mean(features_b, 0)
    sample_cov = np.cov(features_b, rowvar=False)


    dset_a = ImageFolder(root=args.path_a, transform=transform)
    loader_a = iter(DataLoader(dset_a, batch_size=args.batch, num_workers=4, sampler=InfiniteSamplerWrapper(dset_a)))

    features_a = extract_features(loader_a, inception, device).numpy()
    print(f'extracted {features_a.shape[1]} features')

    real_mean = np.mean(features_a, 0)
    real_cov = np.cov(features_a, rowvar=False)

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print(args.epoch, ' fid:', fid)
