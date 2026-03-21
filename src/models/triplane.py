import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""

triplane_network.py
================
文件级说明：
本文件实现了一个基于“三平面（Tri-Plane）”表示的轻量级隐式神经映射网络，专为“条件驱动”的 3D 形状/场景建模设计。  
核心思想：用三个 2D 特征平面（XY、XZ、YZ）来压缩 3D 空间信息，输入任意 3D 坐标即可通过双线性插值快速查询到该点的特征，  
再经一个微型 MLP 解码为所需输出（如 RGB、SDF、偏移量等）。  

对外接口：
1. ModulatedNetwork(**kwargs)  
   ├─ 作用：给定一个条件向量 embd，即时生成三平面特征，并支持对任意 3D 坐标 x 进行前向查询。  
   ├─ 输入：  
   │   - x: [N, 3]  3D 坐标，要求已归一化到 [-1,1]（重要！）  
   │   - embd: [N, embd_dim]  条件向量（例如形状/场景编码）  
   ├─ 输出：  
   │   - out: [N, output_dim]  网络预测值（颜色、密度、位移等，由使用者定义）  
   └─ 特点：  
       - 纯 PyTorch 实现，支持二阶梯度（double backward），可用作可微渲染/优化管线中的隐式场。  
       - 平面分辨率、特征维度、MLP 深度均可通过 __init__ 参数配置。  
       - 初始输出接近零，便于残差式学习。  

2. grid_sample_2d(input, grid, ...)  
   ├─ 作用：手写双线性网格采样，兼容 double backward，用于替换 F.grid_sample。  
   └─ 通常只在 ModulatedNetwork 内部调用，外部无需直接使用。  

使用示例：  
>>> net = ModulatedNetwork(output_dim=3, embd_dim=256).cuda()  
>>> x = torch.rand(1024, 3).cuda() * 2 - 1          # 3D 坐标  
>>> z = torch.randn(1, 256).cuda().expand(1024, -1) # 条件向量  
>>> rgb = net(x, z)                                   # 前向推理  
"""




def grid_sample_2d(input, grid, align_corners=True, padding_mode='zeros'):
    """
    A pure Python implementation of grid_sample to support double backward.
    input: [N, C, H, W]
    grid: [N, H_out, W_out, 2]
    """
    N, C, H, W = input.shape
    N_g, H_out, W_out, _ = grid.shape
    assert N == N_g
    
    if align_corners:
        x = ((grid[..., 0] + 1) / 2) * (W - 1)
        y = ((grid[..., 1] + 1) / 2) * (H - 1)
    else:
        x = ((grid[..., 0] + 1) * W - 1) / 2
        y = ((grid[..., 1] + 1) * H - 1) / 2

    # Get corner pixel coordinates
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # Weights
    wa = (x1.type_as(input) - x) * (y1.type_as(input) - y)
    wb = (x1.type_as(input) - x) * (y - y0.type_as(input))
    wc = (x - x0.type_as(input)) * (y1.type_as(input) - y)
    wd = (x - x0.type_as(input)) * (y - y0.type_as(input))
    
    # Permute input for easier indexing
    input_perm = input.permute(0, 2, 3, 1) # [N, H, W, C]
    
    # Batch indices
    batch_idx = torch.arange(N, device=input.device).view(N, 1, 1).expand(N, H_out, W_out)

    def get_pixel_value(img, x_coord, y_coord):
        # x_coord, y_coord: [N, H_out, W_out]
        
        # Check bounds
        mask = (x_coord >= 0) & (x_coord < W) & (y_coord >= 0) & (y_coord < H)
        
        # Clamp for indexing
        x_cl = x_coord.clamp(0, W - 1)
        y_cl = y_coord.clamp(0, H - 1)
        
        # Fetch
        val = img[batch_idx, y_cl, x_cl] # [N, H_out, W_out, C]
        
        # Apply mask
        return val * mask.unsqueeze(-1).type_as(val)

    Ia = get_pixel_value(input_perm, x0, y0)
    Ib = get_pixel_value(input_perm, x0, y1)
    Ic = get_pixel_value(input_perm, x1, y0)
    Id = get_pixel_value(input_perm, x1, y1)
    
    wa = wa.unsqueeze(-1)
    wb = wb.unsqueeze(-1)
    wc = wc.unsqueeze(-1)
    wd = wd.unsqueeze(-1)
    
    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    
    return out.permute(0, 3, 1, 2) # [N, C, H_out, W_out]

class ModulatedNetwork(nn.Module):
    def __init__(self,
                 input_dim: int = 3,
                 output_dim: int = 3,
                 embd_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 3, # 对于Tri-Plane，解码器可以很浅，3层足够
                 use_pe: bool = False # 这种架构通常不需要PE，或者只需要轻微的PE
                 ) -> None:
        super().__init__()
        
        # --- 配置超参 ---
        # 锚点平面的分辨率。越高越能表示高频细节，但显存占用越大。
        # 64x64 是一个平衡的选择，相当于在空间中撒了 3*64*64 个锚点。
        self.plane_res = 16
        # 每个锚点包含的特征维度
        self.plane_feat_dim = 8
        
        # --- 1. 锚点生成器 (Anchor Generator) ---
        # 它的任务是将 embd "解压" 成三个高维特征平面
        # 输出尺寸: 3个平面 * 特征维 * H * W
        self.total_anchor_params = 3 * self.plane_feat_dim * self.plane_res * self.plane_res
        
        # 使用一个线性层将 condition 映射为密集的锚点参数
        self.anchor_generator = nn.Linear(embd_dim, self.total_anchor_params)

        # --- 2. 轻量级解码器 (Tiny Decoder) ---
        # 负责将插值得到的特征解释为坐标偏移
        # 输入维度 = 3个平面的特征拼接 + 原始坐标(可选)
        in_feat = self.plane_feat_dim * 3 + input_dim
        
        layers = []
        layers.append(nn.Linear(in_feat, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
        
        # 初始化：生成器初始化要小，确保初始平面接近 0
        nn.init.normal_(self.anchor_generator.weight, std=0.001)
        nn.init.zeros_(self.anchor_generator.bias)
        
        # 初始化：解码器最后层初始化为0，保证初始输出接近0（便于学习残差/偏移）
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def generate_planes(self, embd):
        """
        根据条件 embd 生成三个特征平面 (XY, XZ, YZ)
        embd: [Batch, D]
        Return: [Batch, 3, C, H, W]
        """
        B = embd.shape[0]
        
        # 优化：如果 Batch 内所有 embd 是一样的（常见于单物体推理），
        # 我们可以只计算一次以节省显存。这里为了通用性，按标准写法。
        
        # [B, D] -> [B, 3 * C * H * W]
        raw_anchors = self.anchor_generator(embd)
        
        # Reshape 为 [B, 3, C, H, W]
        planes = raw_anchors.view(B, 3, self.plane_feat_dim, self.plane_res, self.plane_res)
        return planes

    def forward(self, x: torch.Tensor, embd: torch.Tensor) -> torch.Tensor:
        """
        x: [N, 3] 输入坐标，假设范围在 [-1, 1] 之间 (这一点非常重要！)
        embd: [N, embd_dim]
        """
        N = x.shape[0]
        
        # 注意：Grid Sample 需要输入在 [-1, 1] 之间。
        # 如果你的输入数据范围不是 [-1, 1]，请务必在这里归一化，或者在外部归一化。
        # 这里为了稳健性，我假设输入可能稍大，用 tanh 软限制一下，或者你可以直接注释掉
        # x_norm = torch.tanh(x) 
        x_norm = x # 假设调用者已经保证 x 在 [-1, 1] 或 [0, 1] 附近
        
        # --- 1. 生成锚点平面 ---
        # 这里有一个 Batch 处理的细节：
        # 如果 N 个点的 embd 都不同，我们需要生成 N 组平面（显存消耗大）。
        # 如果 N 个点的 embd 是一样的（属于同一个物体），我们只需要生成 1 组。
        # 下面的代码处理了最通用的情况（假设 embd 可能不同）。
        
        planes = self.generate_planes(embd) # [N, 3, C, H, W]
        
        # --- 2. 投影与插值 (Look up anchors) ---
        # 我们需要从三个平面采样：
        # XY平面 (取 x, y 坐标)
        # XZ平面 (取 x, z 坐标)
        # YZ平面 (取 y, z 坐标)
        
        # grid_sample 需要坐标格式为 [N, H, W, 2]，值域 [-1, 1]
        # 我们把每个点看作一个 1x1 的“图像”进行采样
        
        # 构造采样坐标: [N, 1, 1, 2]
        sample_xy = x_norm[:, :2].view(N, 1, 1, 2)            # (x, y)
        sample_xz = x_norm[:, [0, 2]].view(N, 1, 1, 2)        # (x, z)
        sample_yz = x_norm[:, [1, 2]].view(N, 1, 1, 2)        # (y, z)
        
        # 分离三个平面 [N, C, H, W]
        plane_xy = planes[:, 0]
        plane_xz = planes[:, 1]
        plane_yz = planes[:, 2]
        
        # 双线性插值 (Bilinear Interpolation)
        # align_corners=True 是 3D 视觉中的标准操作
        # feat_xy = F.grid_sample(plane_xy, sample_xy, align_corners=True) # [N, C, 1, 1]
        # feat_xz = F.grid_sample(plane_xz, sample_xz, align_corners=True)
        # feat_yz = F.grid_sample(plane_yz, sample_yz, align_corners=True)
        
        # 使用自定义的 grid_sample_2d 以支持二阶导数 (Double Backward)
        feat_xy = grid_sample_2d(plane_xy, sample_xy, align_corners=True)
        feat_xz = grid_sample_2d(plane_xz, sample_xz, align_corners=True)
        feat_yz = grid_sample_2d(plane_yz, sample_yz, align_corners=True)
        
        # 展平并拼接 [N, C*3]
        feat = torch.cat([
            feat_xy.view(N, -1), 
            feat_xz.view(N, -1), 
            feat_yz.view(N, -1)
        ], dim=-1)
        
        # 将原始坐标也拼进去，保留低频位置信息
        feat = torch.cat([feat, x], dim=-1)
        
        # --- 3. 解码 (Decode) ---
        out = self.decoder(feat)
        
        return out
