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

        # OPTIMIZATION: Use 1x1 Conv instead of Linear. 
        # This performs the same math but stays in BCHW format. [cite: 4, 8]
        self.fc0 = nn.Conv2d(channel_n * 2, hidden_size, 1)
        self.fc1 = nn.Conv2d(hidden_size, channel_n, 1, bias=False)
        
        padding = (kernel_size - 1) // 2
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1, 
                           padding=padding, padding_mode="reflect", groups=channel_n)
        
        if normalization == "batch":
            self.bn = nn.BatchNorm2d(hidden_size, track_running_stats=False)
        elif normalization == "layer":
            # LayerNorm for 2D is effectively GroupNorm with 1 group [cite: 5]
            self.bn = nn.GroupNorm(1, hidden_size)
        elif normalization == "group":
            self.bn = nn.GroupNorm(1, hidden_size)
        else:
            self.bn = nn.Identity()

        with torch.no_grad():
            self.fc1.weight.zero_()
            if init_method == "xavier":
                nn.init.xavier_uniform_(self.fc0.weight)

        self.to(self.device)

    def update(self, x, fire_rate):
        # All operations here are now natively BCHW [cite: 8]
        y1 = self.p0(x)
        dx = torch.cat((x, y1), 1)
        
        dx = self.fc0(dx)
        dx = self.bn(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
            
        # Stochastic mask: Generated in BCHW to match 
        mask = (torch.rand(x.size(0), 1, x.size(2), x.size(3), device=self.device) > fire_rate).to(x.dtype)
        return dx * mask

    def forward(self, x, steps=10, fire_rate=0.5, visualize: bool=False):
        # 1. Convert to BCHW once at the very beginning [cite: 11]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        gallery = []
        for _ in range(steps):
            dx = self.update(x, fire_rate)
            
            # 2. Optimized persistence: Update only the non-input channels 
            # instead of using torch.concat which re-allocates memory. 
            if self.input_channels > 0:
                x[:, self.input_channels:, ...] += dx[:, self.input_channels:, ...]
            else:
                x += dx
                
            if visualize:
                gallery.append(x.permute(0, 2, 3, 1).detach().cpu())
        
        # 3. Convert back to BHWC once at the end [cite: 10]
        x = x.permute(0, 2, 3, 1)
        return (x, gallery) if visualize else x