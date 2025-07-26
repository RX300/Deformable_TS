import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, AutoEncoderDeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False, use_autoencoder=False, latent_dim=64):
        self.use_autoencoder = use_autoencoder
        self.is_6dof = is_6dof
        
        if use_autoencoder:
            # 使用自编码器变形网络
            self.deform = AutoEncoderDeformNetwork(
                latent_dim=latent_dim,
                is_blender=is_blender, 
                is_6dof=is_6dof
            ).cuda()
        else:
            # 使用原始变形网络
            self.deform = DeformNetwork(
                is_blender=is_blender, 
                is_6dof=is_6dof
            ).cuda()
            
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb, rotation=None, scaling=None):
        """
        如果使用自编码器，需要提供完整的高斯参数
        如果使用原始网络，只需要 xyz
        """
        if self.use_autoencoder:
            if rotation is None or scaling is None:
                raise ValueError("AutoEncoder mode requires rotation and scaling parameters")
            return self.deform(xyz, rotation, scaling, time_emb)
        else:
            return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
