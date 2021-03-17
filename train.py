import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import numpy as np
import argparse
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True
spatial_infogan_size = 4

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


class log_gaussian:

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1) # mul(-1) here is to make it negative Q(c|x) 

criterionQ_con = log_gaussian()

def train_d(net, data, label="real", decode=True, use_infogan=False):
    """Train function of discriminator"""
    if label=="real":
        if decode:
            pred, [rec_all, rec_small, rec_part], part = net(data, label)
            err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
                            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
                            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
                            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = net(data, label)
            err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean()
            err.backward()
            return pred.mean().item()
    else:
        if use_infogan:
            pred, _ = net(data, label)
        else:
            pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    info_lambda = args.info_lambda

    ndf = 64
    ngf = 64
    nz = args.nz # latent dimension
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = not args.use_cpu
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:%d"%(args.cuda))

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    dataset = ImageFolder(root=data_root, transform=trans)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))


    netG = Generator(ngf=ngf, nz=nz, im_size=im_size, sle=(not args.no_sle), \
                    big=args.big, infogan_latent_dim=args.infogan_latent_dim, spatial_code_dim=args.spatial_code_dim)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size, sle=(not args.no_sle), decode=(not args.no_decode), \
                        big=args.big, infogan_latent_dim=args.infogan_latent_dim, spatial_code_dim=args.spatial_code_dim)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    from pytorch_model_summary import summary

    # print(summary( netG, torch.zeros((1, 256+args.infogan_latent_dim+spatial_infogan_size*args.spatial_code_dim)).to(device), show_input=False))
    # print(summary( netD, torch.zeros((1, 3, im_size, im_size)).to(device), 'True', show_input=False))

    avg_param_G = copy_G_params(netG)

    if args.use_infogan:
        fixed_noise = torch.FloatTensor(64, nz).normal_(0, 1)
        latent = torch.FloatTensor(64, args.infogan_latent_dim+spatial_infogan_size*args.spatial_code_dim).uniform_(-1, 1)
        fixed_noise = torch.cat([fixed_noise, latent], dim=1).to(device)
        
        if not args.spatial_code_dim:
            num_steps = 7
            upper = 2
            z = torch.FloatTensor(1, 1, args.infogan_latent_dim).uniform_(-1, 1)
            z = z.expand(args.infogan_latent_dim, num_steps+1, -1).clone()
            intervals = [-upper+i*2*upper/num_steps for i in range(num_steps+1)]
            for i in range(args.infogan_latent_dim):
                for j in range(num_steps+1):
                    z[i, j, i] = intervals[j]
            z = z.reshape(-1, args.infogan_latent_dim)
            fixed_noise_1 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(z.shape[0], 1)
            traversal_z_1 = torch.cat([fixed_noise_1, z], dim=1).to(device)
            fixed_noise_2 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(z.shape[0], 1)
            traversal_z_2 = torch.cat([fixed_noise_2, z], dim=1).to(device)
        else:
            num_steps = 7
            upper = 2

            # same spatial code, traverse latent
            z = torch.FloatTensor(1, 1, args.infogan_latent_dim).uniform_(-1, 1)
            z = z.expand(args.infogan_latent_dim, num_steps+1, -1).clone()
            intervals = [-upper+i*2*upper/num_steps for i in range(num_steps+1)]
            for i in range(args.infogan_latent_dim):
                for j in range(num_steps+1):
                    z[i, j, i] = intervals[j]
            z = z.reshape(-1, args.infogan_latent_dim)

            sz = torch.FloatTensor(1, args.spatial_code_dim).uniform_(-1, 1).repeat(1, spatial_infogan_size)
            sz = sz.repeat(z.shape[0], 1)
            fixed_noise_1 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(z.shape[0], 1)
            traversal_z_1 = torch.cat([fixed_noise_1, z, sz], dim=1).to(device)
            fixed_noise_2 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(z.shape[0], 1)
            traversal_z_2 = torch.cat([fixed_noise_2, z, sz], dim=1).to(device)

            # traverse lower right
            sz = torch.FloatTensor(1, 1, args.spatial_code_dim).uniform_(-1, 1)
            sz = sz.expand(args.spatial_code_dim, num_steps+1, -1).clone()
            for i in range(args.spatial_code_dim):
                for j in range(num_steps+1):
                    sz[i, j, i] = intervals[j]
            # corner latent
            sz = sz.reshape(-1, args.spatial_code_dim)
            # entire latent
            z = torch.FloatTensor(1, args.infogan_latent_dim).uniform_(-1, 1).repeat(sz.shape[0], 1)
            # spatial latent except corner
            sz_all = torch.FloatTensor(1, args.spatial_code_dim*(spatial_infogan_size-1)).uniform_(-1, 1).repeat(sz.shape[0], 1)
            fixed_noise_3 = torch.FloatTensor(1, nz).normal_(0, 1).repeat(sz.shape[0], 1)
            traversal_corner = torch.cat([fixed_noise_3, z, sz_all, sz], dim=1).to(device)

    else:
        fixed_noise = torch.FloatTensor(64, nz).normal_(0, 1).to(device)

    if multi_gpu:
        netG = nn.DataParallel(netG.cuda())
        netD = nn.DataParallel(netD.cuda())

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    if args.use_infogan:
        optimizerQ = optim.Adam([{'params': netD.latent_from_128.parameters()}, \
                                {'params': netD.conv_q.parameters()}], \
                                lr=args.q_lr, betas=(nbeta1, 0.999))
    if args.spatial_code_dim:
        optimizerQ = optim.Adam([{'params': netD.latent_from_128.parameters()}, \
                                {'params': netD.conv_q.parameters()},
                                {'params': netD.spatial_latent_from_128.parameters()}],
                                # {'params': netD.spatial_conv_q.parameters()}], \
                                lr=args.q_lr, betas=(nbeta1, 0.999))
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        if args.use_infogan:
            optimizerQ.load_state_dict(ckpt['opt_q'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
    
    vutils.save_image( next(dataloader).add(1).mul(0.5), saved_image_folder+'/real_image.jpg' )
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)

        if args.use_infogan:
            noise = torch.Tensor(current_batch_size, nz).normal_(0, 1)
            latent = torch.Tensor(current_batch_size, args.infogan_latent_dim+spatial_infogan_size*args.spatial_code_dim).uniform_(-1, 1)
            noise = torch.cat([noise, latent], dim=1).to(device)
            latent = latent.to(device)
        else:
            noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()

    
        err_dr = train_d( netD, real_image, label="real", decode=(not args.no_decode) ) #err on real data backproped
        if not args.no_decode:
            err_dr, rec_img_all, rec_img_small, rec_img_part = err_dr
            

        train_d(netD, [fi.detach() for fi in fake_images], label="fake", use_infogan=args.use_infogan)
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        if args.use_infogan:
            netD.zero_grad()
            pred_g, q_pred = netD(fake_images, "fake")
            if not args.spatial_code_dim:
                q_mu, q_logvar = q_pred[:, :args.infogan_latent_dim], q_pred[:, args.infogan_latent_dim:]
                info_total_loss = criterionQ_con(latent, q_mu, q_logvar.exp())
            else:
                q_pred, s_list = q_pred
                q_mu, q_logvar = q_pred[:, :args.infogan_latent_dim], q_pred[:, args.infogan_latent_dim:]
                info_loss = criterionQ_con(latent[:, :args.infogan_latent_dim], q_mu, q_logvar.exp())
                s_info_loss = 0
                for part in range(4):
                    sq_mu, sq_logvar = s_list[part][:, :args.spatial_code_dim], s_list[part][:, args.spatial_code_dim:]
                    s_info_loss += criterionQ_con(latent[:, \
                        args.infogan_latent_dim+part*args.spatial_code_dim : \
                        args.infogan_latent_dim+(part+1)*args.spatial_code_dim], sq_mu, sq_logvar.exp())

                info_total_loss = s_info_loss/4 + info_loss

            err_g = info_total_loss*args.info_lambda - pred_g.mean()
            err_g.backward()
            optimizerG.step()
            optimizerQ.step()
        else:
            pred_g = netD(fake_images, "fake")
            err_g = -pred_g.mean()

            err_g.backward()
            optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            if args.spatial_code_dim:
                print("GAN: loss d: %.5f    loss g: %.5f    loss info: %.5f    loss s info: %.5f"%(err_dr, -err_g.item(), -info_loss*args.info_lambda, -s_info_loss*args.info_lambda/4))
            elif args.infogan_latent_dim:
                print("GAN: loss d: %.5f    loss g: %.5f    loss info: %.5f"%(err_dr, -err_g.item(), -info_total_loss*args.info_lambda))
            else:
                print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=8)
                if args.use_infogan:
                    vutils.save_image(netG(traversal_z_1)[0].add(1).mul(0.5), saved_image_folder+'/trav1_%d.jpg'%iteration, nrow=num_steps+1)
                    vutils.save_image(netG(traversal_z_2)[0].add(1).mul(0.5), saved_image_folder+'/trav2_%d.jpg'%iteration, nrow=num_steps+1)
                    if args.spatial_code_dim:
                        vutils.save_image(netG(traversal_corner)[0].add(1).mul(0.5), saved_image_folder+'/trav_c_%d.jpg'%iteration, nrow=num_steps+1)
                    
            load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            load_params(netG, backup_para)
            states = {'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}
            if args.use_infogan:
                states['opt_q'] = optimizerQ.state_dict()
            torch.save(states, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='./data/collage/train', help='path of resource dataset, should be a folder that has images (not in sub-folder)')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='new512', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=16, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=512, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path')
    parser.add_argument('--infogan_latent_dim', type=int, default=0, help='infogan latent dim, do not use if = 0')
    parser.add_argument('--spatial_code_dim', type=int, default=0, help='use spatial latent, dimension for each block, do not use if = 0')
    parser.add_argument('--info_lambda', type=float, default=0.1, help='infogan latent dim, do not use if = 0')
    parser.add_argument('--q_lr', type=float, default=0, help='q head learning rate, 0 if backprop through entire D')
    parser.add_argument('--nz', type=int, default=256, help='dimension of noise vector')

    parser.add_argument('--big', dest='big', action="store_true", help='use a more complicated model structure for G and D')
    parser.add_argument('--no_sle', dest='no_sle', action="store_true", help='disable the sle module')
    parser.add_argument('--no_decode', dest='no_decode', action="store_true", help='desable the self-supervised auto-encoding training on Discriminator')
    parser.add_argument('--use_cpu', dest='use_cpu', action="store_true", help='use cpu')
    parser.add_argument('--use_infogan', dest='use_infogan', action="store_true", help='use infogan loss')
    parser.set_defaults(big=False)
    parser.set_defaults(no_sle=False)
    parser.set_defaults(no_decode=False)
    parser.set_defaults(use_cpu=False)
    parser.set_defaults(use_infogan=False)

    args = parser.parse_args()
    print(args)

    train(args)
