import argparse
import os
import time
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from torchvision.models import resnet18
from thop import profile


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='Config.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='ckpt/model_epoch600.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='results/test', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 打印模型参数量
    # print(f"Number of parameters: {count_parameters(diffusion.model)}")

    """
    def cal_torch_model_params(model):
        '''
        :param model:
        :return:
        '''
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'total_params': total_params / 10000, 'total_trainable_params': total_trainable_params / 10000}"""

    print("=> using dataset '{}'".format(config.data.test_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    #_, _, test_loader = DATASET.get_loaders(parse_patches=False)
    test_loader = DATASET.get_testloaders(parse_patches=False)
    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    #计算模型参数量
    #input_shape = (1, 3, 256, 256)
    #diffusion.calculate_model_params_and_flops(input_shape)

    model = DiffusiveRestoration(diffusion, args, config)

    # 计算Flops
    # inputs = torch.randn(1, 3, 256, 256)
    #flops, params = profile(diffusion.model, inputs=(x_cond,))
    #print(flops / 1e9, params / 1e6)  # flops单位G，para单位M

    time_start = time.time()
    model.restore(test_loader)
    time_end = time.time()
    print("spend_time: {:.3f}".format(time_end-time_start))

if __name__ == '__main__':
    main()
