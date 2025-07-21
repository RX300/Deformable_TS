#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import math
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from scene.triangle_model import TriangleModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal


def render(viewpoint_camera, pc : TriangleModel, pipe, bg_color : torch.Tensor, d_xyz : torch.Tensor, d_rotation : torch.Tensor,
            d_scaling : torch.Tensor, use_deform : bool, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_triangles_points[:,0,:].squeeze(), dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda") + 0
    scaling = torch.zeros_like(pc.get_triangles_points[:,0,0].squeeze(), dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()
    density_factor = torch.zeros_like(pc.get_triangles_points[:,0,0].squeeze(), dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = TriangleRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    opacity = pc.get_opacity
    # triangles_points.shape : N*3*3
    triangles_points = pc.get_triangles_points_flatten
    sigma = pc.get_sigma
    num_points_per_triangle = pc.get_num_points_per_triangle
    cumsum_of_points_per_triangle = pc.get_cumsum_of_points_per_triangle
    number_of_points = pc.get_number_of_points
    means2D = screenspace_points

    if use_deform:
        # 结合 d_xyz, d_rotation, d_scaling 对已有的triangles_points进行变形
        # 1. 将四元数 d_rotation 转换为旋转矩阵
        # 归一化四元数以确保是单位四元数
        norm_d_rotation = torch.nn.functional.normalize(d_rotation, p=2, dim=1)
        w, x, y, z = norm_d_rotation[:, 0], norm_d_rotation[:, 1], norm_d_rotation[:, 2], norm_d_rotation[:, 3]
        
        # 计算旋转矩阵
        N = triangles_points.shape[0]
        rotation_matrices = torch.zeros((N, 3, 3), device=triangles_points.device)
        
        rotation_matrices[:, 0, 0] = 1 - 2*y*y - 2*z*z
        rotation_matrices[:, 0, 1] = 2*x*y - 2*z*w
        rotation_matrices[:, 0, 2] = 2*x*z + 2*y*w
        
        rotation_matrices[:, 1, 0] = 2*x*y + 2*z*w
        rotation_matrices[:, 1, 1] = 1 - 2*x*x - 2*z*z
        rotation_matrices[:, 1, 2] = 2*y*z - 2*x*w
        
        rotation_matrices[:, 2, 0] = 2*x*z - 2*y*w
        rotation_matrices[:, 2, 1] = 2*y*z + 2*x*w
        rotation_matrices[:, 2, 2] = 1 - 2*x*x - 2*y*y

        # 2. 计算每个三角形的质心
        centroids = pc.get_triangles_centroids

        # 3. 将顶点移动到以质心为原点的局部坐标系
        triangles_local = pc.get_triangles_points - centroids

        # 4. 应用缩放和旋转
        scaled_triangles = triangles_local * d_scaling.unsqueeze(1)
        rotated_triangles = torch.bmm(scaled_triangles, rotation_matrices)

        # 5. 将顶点移回世界坐标系并应用最终的位移
        triangles_points_deform = rotated_triangles + centroids + d_xyz.unsqueeze(1)

        # 6. 展平
        triangles_points = triangles_points_deform.flatten(0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    else:
        colors_precomp = override_color



    mask = ((torch.sigmoid(pc._mask) > 0.01).float()- torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
    opacity = opacity * mask

    # Rasterize visible triangles to image, obtain their radii (on screen). 
    rendered_image, radii, scaling, density_factor, allmap, max_blending  = rasterizer(
        triangles_points=triangles_points,
        sigma=sigma,
        num_points_per_triangle = num_points_per_triangle,
        cumsum_of_points_per_triangle = cumsum_of_points_per_triangle,
        number_of_points = number_of_points,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        means2D = means2D,
        scaling = scaling,
        density_factor = density_factor
       )

    rets =  {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            "scaling": scaling,
            "density_factor": density_factor,
            "max_blending": max_blending
            }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    #depth_ratio=0
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets





 