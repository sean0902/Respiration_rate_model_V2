import torch
import torch.nn as nn
from DCLS.construct.modules import  Dcls1d 

# With square kernels, equal stride and dilation
class MultiScaleDilationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.dilation_rates = [1, 2, 4, 8]
        self.convs = nn.ModuleList()

        for dilation in self.dilation_rates:
            conv = Dcls1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_count=kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilated_kernel_size=dilation
            )
            self.convs.append(conv)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out_branches = [conv(x) for conv in self.convs]
        out = torch.cat(out_branches, dim=1)
        residual = self.shortcut(x)
        out += residual
        return self.relu(out)

# CNN + Transformer
class RadarBreathingModel(nn.Module):
    def __init__(self, in_channels=11, hidden_size=128, num_transformer_layers=1, num_classes=4):
        super().__init__()
        self.conv_block = nn.Sequential(
            MultiScaleDilationBlock(in_channels, 16, kernel_size=7),
            nn.MaxPool1d(kernel_size=2),
            MultiScaleDilationBlock(16, 32, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            MultiScaleDilationBlock(32, 64, kernel_size=3),
            MultiScaleDilationBlock(64, 128, kernel_size=3),
            MultiScaleDilationBlock(128, 128, kernel_size=3)
        )
        self.feature_projection = nn.Linear(128, hidden_size)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Linear(32, 1)
        )
        # self.moe_head = RangeMoEHead(128, num_experts=5)

        self.reg_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.activity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        x = self.feature_projection(x)
        # pe = get_positional_encoding(x.size(1), x.size(2)).to(x.device)
        # x = x + pe
        x = self.transformer(x)
        x = self.layer_norm(x[:, -1, :])
        x = self.dense(x)
        
        # rr_output = self.moe_head(x)          # (B, 1)
        rr_output = self.reg_head(x)          # (B, 1)
        act_logits = self.activity_head(x)    # (B, num_classes)
        return rr_output, act_logits

# ---------- Multi-scale conv with GroupNorm ----------
class MultiScaleDilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MultiScaleDilationBlock, self).__init__()
        self.dilation_rates = [1, 2, 4, 8]
        self.convs = nn.ModuleList()
        for dilation in self.dilation_rates:
            padding = (kernel_size - 1) * dilation // 2
            conv_branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(self.dilation_rates), kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels // len(self.dilation_rates)),
                nn.ReLU(),
                nn.Conv1d(out_channels // len(self.dilation_rates), out_channels // len(self.dilation_rates), kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels // len(self.dilation_rates))
            )
            self.convs.append(conv_branch)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out_branches = [conv(x) for conv in self.convs]
        out = torch.cat(out_branches, dim=1)
        residual = self.shortcut(x)
        out += residual
        return self.relu(out)