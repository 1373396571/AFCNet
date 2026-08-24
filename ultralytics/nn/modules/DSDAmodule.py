import torch
import torch.nn as nn
import torch.nn.functional as F

class DSDA(nn.Module):
    """使用瓶颈结构减少参数量"""
    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduced_channels = in_channels // reduction_ratio  # 512 → 128


        # 局部分支：512 → 128 → 128 → 512
        self.local_conv1 = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.local_bn1 = nn.GroupNorm(1, self.reduced_channels)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_conv2 = nn.Conv2d(self.reduced_channels, in_channels, 1, bias=False)
        self.local_bn2 = nn.GroupNorm(1, in_channels)
        
        # 全局分支：512 → 128 → 128 → 512
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv1 = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.global_bn1 = nn.GroupNorm(1, self.reduced_channels)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_conv2 = nn.Conv2d(self.reduced_channels, in_channels, 1, bias=False)
        self.global_bn2 = nn.GroupNorm(1, in_channels)
        self.global_sigmoid = nn.Sigmoid()
        # 融合层
        self.fusion_conv = nn.Conv2d(in_channels, self.out_channels, 1, bias=False)
    
    def forward(self, f_G):
        #x1, x2 = f_G
        #x = torch.cat([x1, x2], dim=1)  # [B, 512, H, W]

        if isinstance(f_G, (list, tuple)):
            x1, x2 = f_G
            x = torch.cat([x1, x2], dim=1)
        else:   # 单个张量，沿通道 dim=1 拆成两半
        # 假设张量形状为 [B, 2C, H, W]，对半拆分
            
            x = f_G   # 重新拼接，或做其他运算
        
        # 局部分支
        local_feat = self.local_conv1(x)
        local_feat = self.local_bn1(local_feat)
        local_feat = self.local_relu(local_feat)
        local_feat = self.local_conv2(local_feat)
        local_feat = self.local_bn2(local_feat)
        
        # 全局分支
        global_feat = self.global_avg_pool(x)  # [B, 512, 1, 1]
        global_feat = self.global_conv1(global_feat)
        global_feat = self.global_bn1(global_feat)
        global_feat = self.global_relu(global_feat)
        global_feat = self.global_conv2(global_feat)
        global_feat = self.global_bn2(global_feat)
        global_feat = self.global_sigmoid(global_feat)
        
        # 融合
        weighted_feat = x * global_feat
        fused_feat = weighted_feat + local_feat
        #weighted_feat = x + local_feat
        #fused_feat = weighted_feat * global_feat

        out = self.fusion_conv(fused_feat)
        
        return out