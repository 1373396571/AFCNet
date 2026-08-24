import torch
import torch.nn as nn
import torch.nn.functional as F

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DetailEnhanceGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 浅层特征 → 空间注意力图
        self.spatial_att_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出 [B, 1, H, W]
        )
        # 深层特征 → 通道注意力
        self.channel_att_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C_deep, 1, 1]
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels//2, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出 [B, C_out, 1, 1]
        )
        
    def forward(self, x_deep, skip_shallow):
        c = self.channel_att_gen(x_deep)
        s = self.spatial_att_gen(skip_shallow)
        gate = c*s

        return  gate  # 返回增强特征和门控图(可用于可视化)



class LDC(nn.Module):
    def __init__(self, skip_ch, x_ch, dim):
        super().__init__()
        self.dim = dim
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)  # 初始值0.1

        self.conv_adj_x = nn.Conv2d(x_ch, dim, 1, bias=False) if x_ch != dim else nn.Identity()
        self.conv_adj_skip = nn.Conv2d(skip_ch, dim, 1, bias=False) if skip_ch != dim else nn.Identity()
        

        self.channel_enhance = nn.Sequential(
            nn.Conv2d(dim*2, dim//2, kernel_size=1),
            DyT(dim//2),  
            nn.ReLU(inplace=True),
        )
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dim//2, dim//2, kernel_size=1, bias=False),
            DyT(dim//2), 
            nn.ReLU(inplace=True)
        )
        self.DetailEnhanceGate = DetailEnhanceGate(dim)
        self.mutiscale_conv = nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=3, padding=1, groups=dim//4, bias=False),
            DyT(dim),  
            nn.ReLU(inplace=True)
        )
        

        self.shortcut = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=False),
            DyT(dim)  
        )

        self.final_dy_t = DyT(dim)

    def forward(self, x1):
        skip, x = x1
        
        x = self.conv_adj_x(x)
        
        skip = self.conv_adj_skip(skip)
        

        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        

        channel_enhanced = self.channel_enhance(torch.cat([x, skip], dim=1))
        

        fused = self.fuse_conv(channel_enhanced)

        att = self.DetailEnhanceGate(x,skip)
        

        multiscale = self.mutiscale_conv(fused * att)
        

        residual = self.shortcut(torch.cat([x, skip], dim=1))
        
    
        output = self.final_dy_t(multiscale + residual * self.residual_scale)
        
        return output
    


