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


class GaussianEncoder(nn.Module):
    """编码器：将高斯参数编码到潜在空间"""
    def __init__(self, input_dim=10, latent_dim=64, hidden_dim=128):
        super(GaussianEncoder, self).__init__()
        # input_dim = 3 (xyz) + 4 (rotation) + 3 (scaling) = 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
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
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
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

        self.embed_time_fn, time_input_ch = get_embedder(10, 1)
        self
        self.input_ch = latent_dim + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30
            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 128), nn.LeakyReLU(inplace=True),
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


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, args=None):
        super().__init__()
        self.D = D
        self.W = W
        self.args = args
        self.latent_dim = args.latent_dim
        self.is_blender = args.is_blender
        self.multires = args.multires
        
        # 使用真正的AEDeformable风格的AutoEncoder网络
        self.autoencoder = AutoEncoderDeformNetwork(
            latent_dim=self.latent_dim,
            is_blender=self.is_blender,
            is_6dof=False
        )

    def forward(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb):
        # 提取xyz坐标和处理时间输入
        xyz = rays_pts_emb
        rotations = rotations_emb
        scales = scales_emb
        t = time_emb
        
        # 确保时间维度正确
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.shape[-1] > 1:
            t = t[:, :1]
        
        # 使用真正的AutoEncoder进行变形
        d_xyz, d_rotation, d_scaling = self.autoencoder(xyz, rotations, scales, t)
        
        # 应用变形
        pts = xyz + d_xyz
        scales = scales_emb + d_scaling
        rotations = rotations_emb + d_rotation
        opacity = opacity_emb
        shs = shs_emb
        return pts, scales, rotations, opacity, shs

    def get_mlp_parameters(self):
        return list(self.autoencoder.parameters())

    def compute_regularization_loss(self):
        # 简化版本的正则化损失，兼容现有调用
        reg_loss = 0.0
        for param in self.autoencoder.parameters():
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
        return self.deformation_net(points, scales, rotations, opacity, shs, times)
    
    def get_mlp_parameters(self):
        """获取MLP参数"""
        return self.deformation_net.get_mlp_parameters()
        
    def compute_regularization_loss(self):
        """计算正则化损失"""
        return self.deformation_net.compute_regularization_loss()

class EmbeddingDeformNetwork(nn.Module):
    """在潜在空间中进行时间变形的网络"""
    def __init__(self, embedding_dim, output_dim=10, D=8, W=256, is_blender=False):
        super(EmbeddingDeformNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.D = D
        self.W = W
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.input_ch = embedding_dim + time_input_ch
        
        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30
            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 128), nn.LeakyReLU(inplace=True),
                nn.Linear(128, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(embedding_dim + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + embedding_dim + self.time_out, W)
                    for i in range(D - 1)]
            )
        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender

    def forward(self, embeddings, t):
        """
        embeddings: [N, embeddings_dim] 编码后的高斯参数
        t: [N, 1] 时间输入
        return: [N, latent_dim] 变形后的潜在表示
        """
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)
        
        h = torch.cat([embeddings, t_emb], dim=-1)
        
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                if self.is_blender:
                    h = torch.cat([embeddings, t_emb, h], -1)
                else:
                    h = torch.cat([embeddings, t_emb, h], -1)
        output_embeddings = h  # [N, W]
        return output_embeddings




class deform_embeddingnetwork(nn.Module):
    def __init__(self, D=8, W=256, args=None):
        super(deform_embeddingnetwork, self).__init__()
        self.D = D
        self.W = W
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.is_blender = args.is_blender
        #self.is_6dof = args.is_6dof
        
        # self.embedding_deform = EmbeddingDeformNetwork(
        #     embedding_dim=self.embedding_dim,
        #     output_dim=10,  # 输出维度为10，包含d_xyz(3) + d_rotation(4) + d_scaling(3)
        #     D=self.D,
        #     W=self.W,
        #     is_blender=self.is_blender
        # )

        self.posenc_pos_fn, self.posenc_pos_dim = get_embedder(10, 3)
        self.posenc_time_fn, self.posenc_time_dim = get_embedder(10, 1)
        self.deform_pos_network = self.create_residual_network(self.embedding_dim+self.posenc_time_dim + self.posenc_pos_dim, 3, W)
        self.deform_scale_network = self.create_residual_network(self.embedding_dim+self.posenc_time_dim, 3, W)
        self.deform_rotation_network = self.create_residual_network(self.embedding_dim+self.posenc_time_dim, 4, W)
        self.deform_opacity_network = self.create_residual_network(self.embedding_dim+self.posenc_time_dim, 1, W)
        # self.deform_pos_network = self.create_network(self.embedding_dim+self.posenc_time_dim, 3, W)
        # self.deform_scale_network = self.create_network(self.embedding_dim+self.posenc_time_dim, 3, W)
        # self.deform_rotation_network = self.create_network(self.embedding_dim+self.posenc_time_dim, 4, W)
        # self.deform_opacity_network = self.create_network(self.embedding_dim+self.posenc_time_dim, 1, W)
        #self.deform_rgb_network = self.create_network(W, 3, W)

    @property
    def gaussian_embedding_dim(self):
        return self.embedding_dim

    def create_network(self, input_ch,output_ch,hidden_dim):
        """创建输出网络"""
        return nn.Sequential(nn.Linear(input_ch, hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim, output_ch))

    def create_residual_network(self, input_ch, output_ch, hidden_dim):
        """创建残差网络，支持指定宽度、深度、跳跃层数"""
        
        class ResidualNetwork(nn.Module):
            def __init__(self, input_ch, output_ch, hidden_dim, depth, skip_layers):
                super().__init__()
                self.input_layer = nn.Linear(input_ch, hidden_dim)
                self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth-2)])
                self.output_layer = nn.Linear(hidden_dim, output_ch)
                self.skip_layers = skip_layers
                
            def forward(self, x):
                h = F.leaky_relu(self.input_layer(x))
                residual = h
                for i, layer in enumerate(self.hidden_layers):
                    
                    h = F.leaky_relu(layer(h))
                    # 残差连接
                    if i + 1 in self.skip_layers:
                        h = h + residual
                
                return self.output_layer(h)
        
        return ResidualNetwork(
            input_ch=input_ch, 
            output_ch=output_ch, 
            hidden_dim=hidden_dim,
            depth=4,  # 使用类的深度参数
            skip_layers=[2]  # 中间层添加残差连接
        )
    
    def get_mlp_parameters(self):
        """获取MLP参数"""
        return self.parameters()
    
        # return list(self.embedding_deform.parameters()) + \
        #        list(self.deform_pos_network.parameters()) + \
        #        list(self.deform_scale_network.parameters()) + \
        #        list(self.deform_rotation_network.parameters()) + \
        #        list(self.deform_opacity_network.parameters())
    
    def forward(self, points, embeddings, scales, rotations, opacity, shs, times):
        """
        points: [N, 3] 高斯中心点坐标
        scales: [N, 3] 高斯缩放参数
        rotations: [N, 4] 高斯旋转四元数
        opacity: [N, 1] 高斯透明度
        shs: [N, 9] SH系数
        times: [N, 1] 时间输入
        """
        # # 使用嵌入变形网络进行变形
        # deformed_latent_embeddings = self.embedding_deform(embeddings, times)

        # xyz_emb = self.posenc_pos_fn(points)
        # # contact embeddings with xyz embedding
        # pos_embeddings = torch.cat([embeddings, xyz_emb], dim=-1)
        # scales_embeddings = torch.cat([embeddings, scales], dim=-1)
        # rotations_embeddings = torch.cat([embeddings, rotations], dim=-1)
        # opacity_embeddings = torch.cat([embeddings, opacity], dim=-1)

        time_embeddings = self.posenc_time_fn(times)
        # contact embeddings with time embedding
        embeddings = torch.cat([embeddings, time_embeddings], dim=-1)
        pos_embeddings = self.posenc_pos_fn(points)
        pos_embeddings = torch.cat([embeddings, pos_embeddings], dim=-1)
        # scale_embeddings = torch.cat([embeddings, scales], dim=-1)
        # rotation_embeddings = torch.cat([embeddings, rotations], dim=-1)
        # 应用变形
        d_xyz = self.deform_pos_network(pos_embeddings)
        d_scales = self.deform_scale_network(embeddings)
        d_rotations = self.deform_rotation_network(embeddings)
        d_opacity = self.deform_opacity_network(embeddings)

        points = points + d_xyz
        scales = scales + d_scales
        rotations = rotations + d_rotations
        opacity = opacity + d_opacity

        return points, scales, rotations, opacity, shs