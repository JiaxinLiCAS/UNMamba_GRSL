import torch
import torch.nn.functional as F
from torch import nn
from models.UNMamba_Backbone import UNMambaBackbone
from einops import rearrange
from torchsummary import summary
import math


class UNMambaLinear(nn.Module):
    def __init__(self, num_band, d_model=256, num_endm=3, num_queries_times=30, scale=3.5, ds=4, dropout=5e-2):
        super(UNMambaLinear, self).__init__()
        self.num_band = num_band
        self.num_endm = num_endm
        self.down_ratio = ds
        self.backbone = UNMambaBackbone(in_channels=num_band, hidden_dim=d_model, num_classes=num_endm,
                                        scale=scale, ds=ds, dropout=dropout)

        self.num_queries = num_queries_times * num_endm
        self.query_embed = nn.Embedding(self.num_queries, num_band)
        self.weights = nn.Parameter(torch.ones((num_endm, num_queries_times)))
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        _, _, w, h = x.shape
        # 重构输入空间维度以适应降采样大小
        pad_1 = math.ceil(w / self.down_ratio) * self.down_ratio - w
        pad_2 = math.ceil(h / self.down_ratio) * self.down_ratio - h
        x_patch = F.pad(x, (0, pad_1, 0, pad_2), mode="reflect")
        abun_get = self.backbone(x_patch)
        abun_get = abun_get[:, :, :w, :h]
        endm_get = self.get_endmember()
        recon_linear = torch.einsum('brhw,rl->blhw', [abun_get, endm_get])
        return recon_linear + 1e-7, abun_get, endm_get

    def get_endmember(self):
        query_embed_weight_split = torch.chunk(self.query_embed.weight, self.num_endm, dim=0)
        query_embed_weight_split = torch.stack(query_embed_weight_split)
        endmember_get = self.weights.unsqueeze(-1).repeat(1, 1, self.num_band) * query_embed_weight_split
        endmember_get = torch.mean(endmember_get, dim=1)
        return endmember_get


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_endmember = 4
    num_band = 198
    rows = [100, 100]
    model = UNMambaLinear(num_band).to(device)
    print(model)
    summary(model, [num_band, 100, 100])
    input_data = torch.randn(1, num_band, 100, 100).to(device)
    recon_linear, abun_get, endm_get = model(input_data)
    print(recon_linear.shape)
    print(abun_get.shape)
    print(endm_get.shape)
