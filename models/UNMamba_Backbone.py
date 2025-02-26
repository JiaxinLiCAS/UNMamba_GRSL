import math
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from einops import rearrange


class SumToOne(nn.Module):
    def __init__(self, scale=3.5):
        super(SumToOne, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = torch.softmax(self.scale * x, dim=1)
        return x


class SpeMamba(nn.Module):
    def __init__(self, channels, use_residual=True, ds=4):
        super(SpeMamba, self).__init__()
        self.use_residual = use_residual
        self.down_ratio = ds
        self.ds = ds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up_sample = nn.Upsample(scale_factor=ds, mode='bilinear')
        self.mamba = None

        self.proj = None

    def forward(self, x):
        # 如果模型层还未定义，则根据输入动态设置
        if self.mamba is None or self.proj is None:
            self._build_model(x)
        x_unfolded = x.unfold(2, self.ds, self.ds).unfold(3, self.ds, self.ds)
        # 计算每个块的平均值
        # x_unfolded 的形状为 [B, C, H/16, block_size, W/16, block_size]
        # 我们需要对最后两个维度（即每个块）求平均值
        x_small = x_unfolded.mean(dim=(-1, -2))
        _, _, h, w = x_small.shape
        x_flat = rearrange(x_small, ' b c h w -> b c (h w)')
        x_flat = self.mamba(x_flat)
        x_proj = self.proj(x_flat)
        x_proj = rearrange(x_proj, ' b c (h w) -> b c h w', h=h)
        x_proj = self.up_sample(x_proj)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj

    def _build_model(self, x):
        # 获取输入数据的形状
        batch_size, channels, height, width = x.shape
        self.d_model = int(height/self.down_ratio * width/self.down_ratio)
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to(self.device)

        self.proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU()
        ).to(self.device)


class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=channels,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Linear(channels, channels),
                nn.LayerNorm(channels),
                nn.SiLU()
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)

        if self.use_proj:
            x_flat = self.proj(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()

        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self, channels, use_residual, use_att=True, ds=4):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            # self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels, use_residual=use_residual)
        self.spe_mamba = SpeMamba(channels, use_residual=use_residual, ds=ds)

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(spa_x)
        if self.use_att:
            fusion_x = spa_x * self.weights[0] + spe_x * self.weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class UNMambaBackbone(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True,
                 group_num=4, use_att=True, scale=3.5, ds=4, dropout=5e-2):
        super(UNMambaBackbone, self).__init__()
        self.dropout = dropout
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        self.mamba = nn.Sequential(
            BothMamba(channels=hidden_dim, use_residual=use_residual, use_att=use_att, ds=ds),
            BothMamba(channels=hidden_dim, use_residual=use_residual, use_att=use_att, ds=ds),
            BothMamba(channels=hidden_dim, use_residual=use_residual, use_att=use_att, ds=ds),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            SumToOne(scale=scale),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.mamba(x)
        abun_get = self.cls_head(x)
        if self.training:
            abun_get = F.dropout2d(abun_get, p=self.dropout)
        return abun_get


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, length, dim = 2, 512 * 512, 256
    x = torch.randn(batch, length, dim).to("cuda")
    print(x.shape)
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)
    print(y.shape)

    num_endmember = 3
    num_band = 173
    rows = [100, 100]
    # rows = 100
    model = UNMambaBackbone(in_channels=num_band, num_classes=num_endmember).to(device)
    print(model)
    # summary(model, [num_band, 100, 100])
    input_data = torch.randn(1, num_band, rows[0], rows[1]).to(device)
    pred_abun = model(input_data)
    print(pred_abun.shape)
