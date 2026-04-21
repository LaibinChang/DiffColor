import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.mods import HFRM
from torch.utils.tensorboard import SummaryWriter
from models.gc_arch import GlobalCorrector
from skimage.metrics import peak_signal_noise_ratio as psnr1
from metric.uqim_utils import UIQM
from thop import profile
def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.high_enhance0 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=3, out_channels=64)
        self.high_enhance2 = HFRM(in_channels=3, out_channels=64)
        self.global_correct = GlobalCorrector(in_nc=3, out_nc=3, base_nf=64, cond_nf=32, act_type='relu',
                                              normal01=False)
        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, mean_input_high1, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt, mean_input_high1], dim=1), t)#
            # 计算Flops
            # inputs = torch.randn(1, 3, 256, 256)
            #flops, params = profile(self.Unet, inputs=(torch.cat([x_cond, xt, mean_input_high1], dim=1), t.float()))
            #print(flops / 1e9, params / 1e6)  # flops单位G，para单位M

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

            max_value = torch.max((1 - at).sqrt()/at.sqrt())
            Flag_Globag_Correct = max_value.item()>1.6

            if Flag_Globag_Correct:
                channel_means = torch.mean(xt_next, dim=(2, 3), keepdim=True)
                RGBchannel_means, indices = torch.sort(channel_means, dim=1)

                sorted_indices = torch.argsort(channel_means.view(n, c), dim=1)
                indices_for_tensor2 = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
                Reorder_xt_next = torch.gather(xt_next, 1, indices_for_tensor2)

                Schannel_means = RGBchannel_means[:, 0, :, :]
                Mchannel_means = RGBchannel_means[:, 1, :, :]
                Lchannel_means = RGBchannel_means[:, 2, :, :]

                Schannelxt_next = Reorder_xt_next[:, 0, :, :]
                Mchannelxt_next = Reorder_xt_next[:, 1, :, :]
                Lchannelxt_next = Reorder_xt_next[:, 2, :, :]

                Lchannelxt_nextCR = Lchannelxt_next
                Mchannelxt_nextCR = Mchannelxt_next + (Lchannel_means-Mchannel_means)*Lchannelxt_next
                Schannelxt_nextCR = Schannelxt_next + (Lchannel_means - Schannel_means) * Lchannelxt_next

                GCB_xt_next = torch.stack((Schannelxt_nextCR, Mchannelxt_nextCR, Lchannelxt_nextCR), dim=1)

                # Global Color Balance
                Gxt_next = self.global_correct(GCB_xt_next, t)
                #Gxt_next = self.global_correct(xt_next, t)
            else:
                Gxt_next = xt_next
            xs.append(Gxt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        input_high0 = self.high_enhance0(input_high0)

        input_LL_dwt = dwt(input_LL)
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]

        input_high1 = self.high_enhance1(input_high1)

        #第三次小波变换
        input_LL_LL_dwt = dwt(input_LL_LL)
        input_LL_LL_LL, input_high2 = input_LL_LL_dwt[:n, ...], input_LL_LL_dwt[n:, ...]

        input_high2 = self.high_enhance2(input_high2)

        # 三个增强后细节分量，对应元素求均值
        b, c, h, w = input_high2.shape
        input_high2_HL, input_high2_LH, input_high2_HH = input_high2[:b // 3, ...], input_high2[b // 3:2 * b // 3,
                                                                                    ...], input_high2[2 * b // 3:, ...]

        mean_input_high2 = (input_high2_HL + input_high2_LH + input_high2_HH) / 3.0



        """
        #三个增强后细节分量，对应元素求均值
        b, c, h, w = input_high1.shape
        input_high1_HL, input_high1_LH, input_high1_HH = input_high1[:b // 3, ...], input_high1[b // 3:2 * b // 3, ...], input_high1[2 * b // 3:, ...]

        mean_input_high1 = (input_high1_HL+input_high1_LH+input_high1_HH)/3.0"""

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        #Gaussian noise, Size is same with input_LL_LL
        e = torch.randn_like(input_LL_LL_LL)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            gt_LL_LL_dwt = dwt(gt_LL_LL)
            gt_LL_LL_LL, gt_high2 = gt_LL_LL_dwt[:n, ...], gt_LL_LL_dwt[n:, ...]

            x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_LL_LL_LL, x, mean_input_high2], dim=1), t.float())
            denoise_LL_LL_LL = self.sample_training(input_LL_LL_LL, b, mean_input_high2)
            pred_LL_LL = idwt(torch.cat((denoise_LL_LL_LL, input_high2), dim=0))

            pred_LL = idwt(torch.cat((pred_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["input_high2"] = input_high2
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["gt_high2"] = gt_high2
            data_dict["pred_LL_LL"] = pred_LL_LL
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL_LL"] = gt_LL_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL_LL, b, mean_input_high2)
            pred_LL_LL = idwt(torch.cat((denoise_LL_LL, input_high2), dim=0))
            pred_LL = idwt(torch.cat((pred_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0


    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        self.start_epoch=checkpoint['epoch']
        self.step=checkpoint['step']
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader, _ = DATASET.get_loaders()
        writer = SummaryWriter('logs')
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        #interrupt starting setting
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            #print('epoch: ', epoch)
            print("epoch: {}/{}".format(epoch, self.config.training.n_epochs))
            data_start = time.time()
            data_time = 0
            self.step = 0
            total_batches = len(train_loader)
            #batch_size = train_loader.batch_size
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1
                x = x.to(self.device)
                output = self.model(x)
                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)

                loss = noise_loss + photo_loss + frequency_loss
                #if self.step % 10 == 0:
                print("step:{}/{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                          "frequency_loss:{:.4f}".format(self.step, total_batches, self.scheduler.get_last_lr()[0],
                                                         noise_loss.item(), photo_loss.item(),
                                                         frequency_loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
            # 将损失和精度信息写入 TensorBoard
            writer.add_scalar('Loss/lr', self.scheduler.get_last_lr()[0], global_step=epoch)
            writer.add_scalar('Loss/noise_loss', noise_loss.item(), global_step=epoch)
            writer.add_scalar('Loss/photo_loss', photo_loss.item(), global_step=epoch)
            writer.add_scalar('Loss/frequency_loss', frequency_loss.item(), global_step=epoch)
            #if self.step % self.config.training.validation_freq == 0 and self.step != 0:
            if (epoch+1) % self.config.training.val_epochs == 0 and epoch != 0:
                self.model.eval()
                #self.sample_validation_patches(val_loader, self.step, epoch+1)
                mssim, mpsnr, muiqm = self.sample_validation_patches(val_loader, self.step, epoch+1)
                writer.add_scalar('Loss/mssim', mssim, global_step=epoch)
                writer.add_scalar('Loss/mpsnr', mpsnr, global_step=epoch)
                writer.add_scalar('Loss/muiqm', muiqm, global_step=epoch)

            if (epoch+1) % self.config.training.save_epochs == 0 and epoch != 0 and (epoch+1) >199:

                utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                   #filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))
                                                    filename=os.path.join(self.config.data.ckpt_dir, 'model_epoch{}'.format(epoch+1)))
            self.scheduler.step()
        # 关闭 TensorBoard writer
        writer.close()
    def estimation_loss(self, x, output):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)
        # =============frequency loss==================
        frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                self.l2_loss(input_high1, gt_high1) +
                                self.l2_loss(pred_LL, gt_LL)) +\
                         0.01 * (self.TV_loss(input_high0) +
                                 self.TV_loss(input_high1) +
                                 self.TV_loss(pred_LL))
        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        photo_loss = (content_loss + ssim_loss)
        return noise_loss, photo_loss, frequency_loss
    
    def Tensor_Numpy(self, Tensor_input):
        Tensor_input = Tensor_input.cpu().clone()
        Tensor_input = Tensor_input.squeeze(0)  # remove the fake batch dimension
        Tensor_input = Tensor_input.permute(1, 2, 0)
        Numpy_out = Tensor_input.numpy()
        Numpy_out = (Numpy_out * 255).astype(np.uint8)
        return Numpy_out
    
    def sample_validation_patches(self, val_loader, step, epoch):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            sumssim, sumpsnr, sumuiqm = 0., 0., 0.
            N = 0
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                gt_img = x[:, 3:, :, :].to(self.device)

                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]

                #计算SSIM
                ssim_value = ssim(pred_x, gt_img, data_range=1.0).to(self.device)
                #计算PSNR
                gt_img_numpy = self.Tensor_Numpy(gt_img)
                pred_x_numpy = self.Tensor_Numpy(pred_x)
                psnr_value = psnr1(gt_img_numpy, pred_x_numpy)
                #计算UIQM
                uiqm_value = UIQM(pred_x_numpy)

                sumssim += ssim_value.item()
                sumpsnr += psnr_value
                sumuiqm += uiqm_value
                N += 1
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(epoch), f"{y[0]}.jpg"))

            mssim = sumssim/N
            mpsnr = sumpsnr/N
            muiqm = sumuiqm/N
            #muciqe = sumuciqe/N
        return mssim, mpsnr, muiqm