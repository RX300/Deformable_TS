import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class GaussianEncoder(nn.Module):
    """编码器：将高斯参数编码到潜在空间"""
    def __init__(self, input_dim=10, latent_dim=64, hidden_dim=128):
        super(GaussianEncoder, self).__init__()
        # input_dim = 3 (xyz) + 4 (rotation) + 3 (scaling) = 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, gaussian_params):
        """
        gaussian_params: [N, 10] 包含 xyz(3) + rotation(4) + scaling(3)
        """
        return self.encoder(gaussian_params)


class GaussianDecoder(nn.Module):
    """解码器：将潜在表示解码回高斯参数变化"""
    def __init__(self, latent_dim=64, output_dim=10, hidden_dim=128):
        super(GaussianDecoder, self).__init__()
        # output_dim = 3 (d_xyz) + 4 (d_rotation) + 3 (d_scaling) = 10
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, latent_code):
        """
        latent_code: [N, latent_dim]
        return: [N, 10] 包含 d_xyz(3) + d_rotation(4) + d_scaling(3)
        """
        return self.decoder(latent_code)


class LatentDeformNetwork(nn.Module):
    """在潜在空间中进行时间变形的网络"""
    def __init__(self, latent_dim=64, D=6, W=128, multires=6, is_blender=False):
        super(LatentDeformNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.D = D
        self.W = W
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.input_ch = latent_dim + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30
            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 128), nn.ReLU(inplace=True),
                nn.Linear(128, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(latent_dim + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + latent_dim + self.time_out, W)
                    for i in range(D - 1)]
            )
        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.output_layer = nn.Linear(W, latent_dim)

    def forward(self, latent_code, t):
        """
        latent_code: [N, latent_dim] 编码后的高斯参数
        t: [N, 1] 时间输入
        return: [N, latent_dim] 变形后的潜在表示
        """
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)
        
        h = torch.cat([latent_code, t_emb], dim=-1)
        
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                if self.is_blender:
                    h = torch.cat([latent_code, t_emb, h], -1)
                else:
                    h = torch.cat([latent_code, t_emb, h], -1)

        delta_latent = self.output_layer(h)
        return latent_code + delta_latent  # 残差连接


class DeformNetwork(nn.Module):
    """原始的变形网络，保持向后兼容"""
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling


class AutoEncoderDeformNetwork(nn.Module):
    """自编码器变形网络：编码->变形->解码"""
    def __init__(self, latent_dim=64, is_blender=False, is_6dof=False):
        super(AutoEncoderDeformNetwork, self).__init__()
        
        self.latent_dim = latent_dim
        self.is_6dof = is_6dof
        
        # 编码器：将高斯参数编码到潜在空间
        self.encoder = GaussianEncoder(input_dim=10, latent_dim=latent_dim)
        
        # 潜在空间变形网络
        self.latent_deform = LatentDeformNetwork(
            latent_dim=latent_dim, 
            is_blender=is_blender
        )
        
        # 解码器：将潜在表示解码回参数变化
        self.decoder = GaussianDecoder(latent_dim=latent_dim, output_dim=10)
        
        # 正则化项权重
        self.recon_weight = 1.0
        self.latent_reg_weight = 0.01
        
    def forward(self, xyz, rotation, scaling, t):
        """
        xyz: [N, 3] 高斯中心点坐标
        rotation: [N, 4] 高斯旋转四元数
        scaling: [N, 3] 高斯缩放参数
        t: [N, 1] 时间输入
        """
        # 1. 将高斯参数拼接
        gaussian_params = torch.cat([xyz, rotation, scaling], dim=-1)  # [N, 10]
        
        # 2. 编码到潜在空间
        latent_code = self.encoder(gaussian_params)  # [N, latent_dim]
        
        # 3. 在潜在空间中进行时间变形
        deformed_latent = self.latent_deform(latent_code, t)  # [N, latent_dim]
        
        # 4. 解码回参数变化
        delta_params = self.decoder(deformed_latent)  # [N, 10]
        
        # 5. 分离出各个变化量
        d_xyz = delta_params[:, :3]
        d_rotation = delta_params[:, 3:7]
        d_scaling = delta_params[:, 7:10]
        
        return d_xyz, d_rotation, d_scaling
    
    def compute_regularization_loss(self, xyz, rotation, scaling, t):
        """计算正则化损失"""
        # 1. 编码
        gaussian_params = torch.cat([xyz, rotation, scaling], dim=-1)
        latent_code = self.encoder(gaussian_params)
        
        # 2. 潜在空间正则化 - 鼓励潜在代码接近标准正态分布
        latent_reg_loss = torch.mean(latent_code ** 2) * self.latent_reg_weight
        
        # 3. 重建损失 - 确保编码解码一致性
        reconstructed_params = self.decoder(latent_code)
        recon_loss = F.mse_loss(reconstructed_params, torch.zeros_like(reconstructed_params)) * self.recon_weight
        
        return latent_reg_loss + recon_loss
