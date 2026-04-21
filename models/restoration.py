import torch
import numpy as np
import utils
import os
import torch.nn.functional as F
from thop import profile


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, test_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.test_dataset)
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)

                x_output = x_output[:, :, :h, :w]
                # add for sparing the output results
                for k in range(x_output.shape[0]):
                    x_output_single = x_output[k]

                    utils.logging.save_image(x_output_single, os.path.join(image_folder, f"{y[k]}.jpg"))
                    print(f'processing image {y[k]}')

                # utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
                # print(f"processing image {y[0]}")

    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]