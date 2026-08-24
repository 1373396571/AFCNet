import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class FSE(nn.Module):
    def __init__(self, in_channels, number_K, lamda): 
        super().__init__()

        self.spatial_net = nn.Sequential(*[nn.Conv2d(in_channels, in_channels//2, 3, 1, 1, groups=1), 
                                        nn.BatchNorm2d(in_channels//2),
                                        nn.LeakyReLU(),
                                        ])
        self.spectral_net = nn.Sequential(*[nn.Conv2d(in_channels, in_channels//2, 3, 1, 1, groups=1), 
                                        nn.BatchNorm2d(in_channels//2),
                                        nn.LeakyReLU(),
                                        ])
        self.fuse_net = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.LeakyReLU(),
                                    ])
        self.fim = LearnableFreFilter(number_K, lamda)

        # 初始化分支
        self.lfu = LFU_LowFreqUnit(in_channels)

    def forward(self, x):
        feat_f = self.fim(x)
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)

        feat_low = self.lfu(x_out)

        return feat_low


class RadialBasisFilter(nn.Module):
    def __init__(self, n_coeff, lamda, butter_order=2):
        super().__init__()
        self.n_coeff = n_coeff        # 径向基数量K，默认10，
        self.n_ang_freq = 1            # 角度调制谐波数
        self.butter_order = butter_order# 巴特沃斯阶数n，推荐2/4，n越大过渡带越陡
        self.lamda = lamda              # 预留参数

        # 1. 核心可学习参数（和原代码逻辑完全对齐，训练无需改学习率）
        self.coeff_mag   = nn.Parameter(torch.zeros(n_coeff))   # 对应公式a_k，幅度加权系数
        self.coeff_phase = nn.Parameter(torch.zeros(n_coeff))   # 相位加权系数
        self.raw_gate_mag = nn.Parameter(torch.ones(n_coeff))   # 幅度门控，sigmoid后0~1
        self.raw_gate_phase = nn.Parameter(torch.ones(n_coeff)) # 相位门控

        # 2. 中心频率μ_k：0~1线性等距分布
        mu = torch.linspace(0.0, 1.0, steps=n_coeff)
        self.register_buffer('mu', mu)

        # 3. 可学习带宽Δμ：对应原代码的σ_h，控制每个基的通带宽度
        self.log_bandwidth = nn.Parameter(torch.tensor(0.0)) # 用log保证带宽始终为正

    def forward(self, H: int, W: int, device, dtype):
        # 1. 生成频域坐标（和原代码完全一致，无任何改动）
        fy = torch.fft.fftfreq(H, dtype=dtype, device=device)[:, None]
        fx = torch.fft.rfftfreq(W, dtype=dtype, device=device)[None, :]
        r_hat = torch.sqrt(fx ** 2 + fy ** 2)
        r_hat = r_hat / r_hat.max()  # 归一化到0~1

        # 2. 生成巴特沃斯径向基
        bandwidth = torch.exp(self.log_bandwidth) + 1e-6  # 保证带宽为正，避免除0
        # 扩展维度，适配批量计算 [K, H, W/2+1]
        r_expand = r_hat.unsqueeze(0)          # [1, H, W/2+1]
        mu_expand = self.mu[:, None, None]      # [K, 1, 1]

        # 巴特沃斯带通基核心计算（对应上面的数学公式）
        numerator = r_expand ** 2 - mu_expand ** 2  # 分子：r² - μ_k²
        denominator = r_expand * bandwidth + 1e-8    # 分母：r·Δμ，加epsilon避免除0
        butter_term = (numerator / denominator) ** (2 * self.butter_order)
        basis = 1.0 / torch.sqrt(1.0 + butter_term)  # 最终巴特沃斯径向基

        # 3. 门控控制,实现单频段硬/软抑制
        gate_mag = torch.sigmoid(self.raw_gate_mag)[:, None, None]
        gate_phase = torch.sigmoid(self.raw_gate_phase)[:, None, None]

        # 4. 角度调制
        angular_mod = 0
        theta = torch.atan2(fy, fx + 1e-8)
        for n in range(1, self.n_ang_freq + 1):
            angular_mod += torch.cos(n * theta) + torch.sin(n * theta)
        angular_mod = angular_mod / (2 * self.n_ang_freq)
        angular_mod = 1 + self.lamda * angular_mod  # lamda论文默认0.1
        basis = basis * angular_mod.unsqueeze(0)    # 基函数×角度调制

        # 5. 加权求和输出
        diff_mag = (gate_mag * self.coeff_mag[:, None, None] * basis).sum(0, keepdim=True)
        diff_phase = (gate_phase * self.coeff_phase[:, None, None] * basis).sum(0, keepdim=True)
        return diff_mag, diff_phase

class LearnableFreFilter(nn.Module):
    def __init__(self, number_K = 10, lamda = 0.1):
        super().__init__()
        def generate_random_number():
            number = round(random.uniform(0.95, 1.05), 2)
            return number
        
        self.init_sigma_ratio = 0.2

        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self.rad_filter  = RadialBasisFilter(number_K, lamda)

        self._sigma_init = False
    # ------------------------------------------------------------------
    def forward(self, x):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device

        if not self._sigma_init:
            sigma_px = self.init_sigma_ratio * min(H, W)
            with torch.no_grad():
                self.log_sigma.copy_(torch.tensor(np.log(sigma_px), dtype=dtype, device=device))
            self._sigma_init = True

        diff_mag, diff_phase = self.rad_filter(H, W, device, dtype)  # (1,H,W/2+1)
        D = diff_mag.to(dtype) * torch.exp(1j * diff_phase.to(dtype))

        fy = torch.fft.fftfreq(H, dtype=dtype, device=device)[:, None]
        fx = torch.fft.rfftfreq(W, dtype=dtype, device=device)[None, :]
        r_grid = torch.sqrt(fx ** 2 + fy ** 2)
        sigma = torch.exp(self.log_sigma)
        Wg = torch.exp(- (r_grid / sigma) ** 2)  
        Wg = Wg.clone(); Wg[0, 0] = 0.0      

        fft_x = torch.fft.rfft2(x, norm='ortho')   
        fccr_feat = torch.fft.irfft2((D * Wg) * fft_x, s=(H, W), norm='ortho') 

        return fccr_feat


class LFU_LowFreqUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels,in_channels,7,1,3, groups=in_channels, bias=False),
                                     nn.BatchNorm2d(in_channels))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1, groups=in_channels, bias=False),
                                     nn.BatchNorm2d(in_channels))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,2,dilation=2, groups=in_channels, bias=False),
                                     nn.BatchNorm2d(in_channels))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,3,dilation=3, groups=in_channels, bias=False),
                                     nn.BatchNorm2d(in_channels))
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 可学习的缩放因子
        self.channel_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        feat_mid = b1 + b2 + b3 + b4
        ca_weight = self.channel_mixer(feat_mid)
        out = x * ca_weight #+ feat_mid * self.gamma
        return out
   
   
   
