import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AutoEncoderDeformNetwork(nn.Module):
    """完整的自编码器变形网络，与AEDeformable保持一致"""
    def __init__(self, D=8, W=256, multires=10, latent_dim=64, is_blender=False):
        super(AutoEncoderDeformNetwork, self).__init__()
        
        self.D = D
        self.W = W
        self.latent_dim = latent_dim
        self.is_blender = is_blender
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        # 位置编码
        self.embed_fn, self.xyz_input_ch = get_embedder(multires, 3)
        # 时间编码
        self.embed_time_fn, self.time_input_ch = get_embedder(self.t_multires, 1)
        
        # 输入维度 = 位置编码 + 时间编码
        self.input_ch = self.xyz_input_ch + self.time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30
            self.timenet = nn.Sequential(
                nn.Linear(self.time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(self.xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )
        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        # 输出层，预测位置、旋转、缩放的变化
        self.gaussian_warp = nn.Linear(W, 3)     # 位置变化
        self.gaussian_rotation = nn.Linear(W, 4) # 旋转变化
        self.gaussian_scaling = nn.Linear(W, 3)  # 缩放变化
        
    def forward(self, xyz, time_input):
        """
        xyz: [N, 3] 高斯中心点坐标
        time_input: [N, 1] 时间输入
        返回: d_xyz, d_rotation, d_scaling
        """
        # 位置编码
        x_emb = self.embed_fn(xyz)
        # 时间编码
        t_emb = self.embed_time_fn(time_input)
        
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
            h = torch.cat([x_emb, t_emb], dim=-1)
        else:
            h = torch.cat([x_emb, t_emb], dim=-1)
        
        # MLP前向传播
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                if self.is_blender:
                    h = torch.cat([x_emb, t_emb, h], -1)
                else:
                    h = torch.cat([x_emb, t_emb, h], -1)

        # 输出变形参数
        d_xyz = self.gaussian_warp(h)
        d_rotation = self.gaussian_rotation(h)
        d_scaling = self.gaussian_scaling(h)
        
        return d_xyz, d_rotation, d_scaling


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.args = args
        
        # 直接使用参数，不再做兼容性检查
        self.latent_dim = args.latent_dim
        self.is_blender = args.is_blender
        self.multires = args.multires
        
        # 核心AutoEncoder变形网络
        self.autoencoder_deform = AutoEncoderDeformNetwork(
            D=D,
            W=W,
            multires=self.multires,
            latent_dim=self.latent_dim,
            is_blender=self.is_blender
        )
        
    def query_time(self, position, time_input):
        """简化的时间查询接口"""
        # 处理时间输入维度
        if time_input.dim() == 1:
            time_input = time_input.unsqueeze(-1)
        elif time_input.shape[-1] > 1:
            time_input = time_input[:, :1]
        
        # 直接通过AutoEncoder获取变形
        return self.autoencoder_deform(position, time_input)
        
    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb):
        """优化的动态变形"""
        position = rays_pts_emb[:, :3]
        
        # 获取变形量
        d_xyz, d_rotation, d_scaling = self.query_time(position, time_emb)
        
        # 应用变形 - 直接增量，不做条件判断
        pts = position + d_xyz
        scales = scales_emb[:, :3] + d_scaling
        rotations = rotations_emb[:, :4] + d_rotation
        
        # opacity和shs直接透传
        opacity = opacity_emb[:, :1]
        shs = shs_emb
        
        return pts, scales, rotations, opacity, shs
    
    def forward(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb):
        """简化的前向传播"""
        return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb)
    
    def get_mlp_parameters(self):
        """获取所有网络参数"""
        return list(self.autoencoder_deform.parameters())
        
    def compute_regularization_loss(self):
        """简化的正则化损失 - 基于网络权重"""
        reg_loss = 0.0
        for param in self.autoencoder_deform.parameters():
            reg_loss += torch.sum(param ** 2)
        return reg_loss * 1e-6
class deform_network(nn.Module):
    """优化后的变形网络 - 纯AutoEncoder架构"""
    def __init__(self, args):
        super(deform_network, self).__init__()
        
        # 核心变形网络
        self.deformation_net = Deformation(
            W=args.net_width, 
            D=args.defor_depth, 
            args=args
        )
        
        # 只保留必要的初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """简化的权重初始化"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, points, scales, rotations, opacity, shs, times):
        """简化的前向传播 - 直接处理原始参数"""
        return self.deformation_net(
            points.unsqueeze(1) if points.dim() == 2 else points,  # 确保维度正确
            scales.unsqueeze(1) if scales.dim() == 2 else scales,
            rotations.unsqueeze(1) if rotations.dim() == 2 else rotations,
            opacity.unsqueeze(1) if opacity.dim() == 2 else opacity,
            shs.unsqueeze(1) if shs.dim() == 2 else shs,
            times
        )
    
    def get_mlp_parameters(self):
        """获取MLP参数"""
        return self.deformation_net.get_mlp_parameters()
        
    def compute_regularization_loss(self):
        """计算正则化损失"""
        return self.deformation_net.compute_regularization_loss()

