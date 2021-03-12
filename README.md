# collage-gan

To train a FastGAN model, run
```bash
$ python train.py --im_size 256 --iter 50000 --name exp1
```

To train with infogan loss, include the following additional tags with your choice of hyperparameters:
- `--use_infogan`
- `--q_lr`: must specify a positive number, suggested using 0.0001 
- `--info_lambda`: must specify a positive number, such as 0.1, 1, 10, etc. 
- `--infogan_latent_dim`: must specify a positive integar

To train with spatial infogan loss, include the following additional tag:
- `--spatial_code_dim`: must specify a positive integar

Sampled images and latent traversal results are saved during training.
To visualize other results, run 
```bash
$ python visualize.py
```
with the following tags:
- `--dir_name`
- `--ckpt_iter`
- `--method`: one of the following: latent_traversal (traversing the latent code in infogan loss case), sample_fixed_latent (fix latent code and sample noise vectors), sample_fixed_noise (fix noise vector and sample latent code), interpolation (interpolate between two random noise)
- `--num_samples`: number of samples to generate for method "sample_fixed_latent" and "sample_fixed_noise"
- `--num_steps`: number of steps to take in latent traversal or interpolation
