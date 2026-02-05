###############################################
##########   Gemini version   #################
###############################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNCA2D(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=9, groups=False,
                 normalization="batch"):
        super(BasicNCA2D, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.fire_rate = fire_rate

        # Use 1x1 Convolutions instead of Linear layers to avoid transposing BHWC <-> BCHW
        self.fc0 = nn.Conv2d(channel_n * 2, hidden_size, kernel_size=1)
        self.fc1 = nn.Conv2d(hidden_size, channel_n, kernel_size=1, bias=False)
        
        padding = (kernel_size - 1) // 2
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1, 
                           padding=padding, padding_mode="reflect", groups=channel_n)
        
        if normalization == "batch":
            self.bn = nn.BatchNorm2d(hidden_size, track_running_stats=False)
        elif normalization == "layer":
            # LayerNorm over channels requires specific shape or 1D input
            self.bn = nn.GroupNorm(1, hidden_size) # GroupNorm with 1 group is equivalent to LayerNorm for 2D
        elif normalization == "group":
            self.bn = nn.GroupNorm(1, hidden_size)
        elif normalization == "none":
            self.bn = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type {normalization}")

        with torch.no_grad():
            self.fc1.weight.zero_()
            if init_method == "xavier":
                nn.init.xavier_uniform_(self.fc0.weight)
                nn.init.xavier_uniform_(self.fc1.weight)

        self.to(self.device)

    def perceive(self, x):
        # x is BCHW
        y1 = self.p0(x)
        return torch.cat((x, y1), 1)

    def update(self, x, fire_rate):
        # Operates entirely in BCHW format to maximize speed
        dx = self.perceive(x)
        dx = self.fc0(dx)
        dx = self.bn(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        
        # Stochastic mask in BCHW
        mask = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=self.device) > fire_rate
        dx = dx * mask.float()

        return x + dx

    def forward(self, x, steps=10, fire_rate=0.5, visualize: bool=False):
        # Expects input x as BHWC, converts to BCHW once
        x = x.permute(0, 3, 1, 2).contiguous()
        
        gallery = []
        for _ in range(steps):
            x2 = self.update(x, fire_rate)
            
            # Keep input channels persistent while updating NCA state channels
            if self.input_channels > 0:
                x = torch.cat((x[:, :self.input_channels, ...], x2[:, self.input_channels:, ...]), 1)
            else:
                x = x2
                
            if visualize:
                gallery.append(x.permute(0, 2, 3, 1).detach().cpu())
        
        # Convert back to BHWC for the final output
        x_out = x.permute(0, 2, 3, 1)
        
        if visualize:
            return x_out, gallery
        return x_out