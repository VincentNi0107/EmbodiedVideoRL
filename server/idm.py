import torch
import torch.nn as nn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = None
        self.conv = doubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        if self.up is None:
            self.up = nn.Upsample(size=x2.size()[2:], mode='bilinear', align_corners=True)
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


def doubleConv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def down(in_channels,out_channels):
    layer = []
    layer.append(nn.MaxPool2d(2,stride=2))
    layer.append(doubleConv(in_channels, out_channels))
    return nn.Sequential(*layer)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channel=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = 5
        self.in_conv = doubleConv(self.in_channels, base_channel)
        self.downs = [None] * self.num_layers
        self.ups = [None] * self.num_layers
        for i in range(self.num_layers):
            down_in_channel = base_channel * 2 ** i
            down_out_channel = down_in_channel * 2 if i < self.num_layers - 1 else down_in_channel
            up_in_channel = base_channel * 2 ** (self.num_layers - i)
            up_out_channel = up_in_channel // 4 if i < self.num_layers - 1 else base_channel
            self.downs[i] = down(down_in_channel, down_out_channel)
            self.ups[i] = Up(up_in_channel, up_out_channel)
        self.downs = nn.Sequential(*self.downs)
        self.ups = nn.Sequential(*self.ups)
        self.out = nn.Conv2d(in_channels=base_channel, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        xs = [x]
        for down in self.downs:
            x = down(xs[-1])
            xs.append(x)

        x_out = None
        for x, up in zip(xs[::-1][1:], self.ups):
            if x_out is None:
                x_out = up(xs[-1], xs[-2])
            else:
                x_out = up(x_out, x)
        out = self.out(x_out)
        return out


class BottleNeckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, output_dim=14, input_channels=3, *args, **kwargs):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.conv2_x = self._make_layer(64, 3, 1)
        self.conv3_x = self._make_layer(128, 4, 2)
        self.conv4_x = self._make_layer(256, 6, 2)
        self.conv5_x = self._make_layer(512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, output_dim)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BottleNeckResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class GoalConditionedMask(nn.Module):
    """Dual-encoder IDM: takes (observation, goal_frame) and predicts actions.

    The observation provides current robot state; the goal frame (from a
    generated video) serves as a visual target.  A shared UNet produces
    separate attention masks for each input, then two ResNet50 encoders
    extract features that are concatenated and mapped to the 14-DOF action.
    """

    def __init__(self, output_dim: int = 14):
        super().__init__()
        self.output_dim = output_dim

        # 6-channel input (obs + goal) → 2-channel mask (one per input)
        self.mask_model = UNet(6, 2)

        # Separate encoders — will be init'd from the same pretrained weights
        self.obs_encoder = ResNet(output_dim=2048, input_channels=3)
        self.goal_encoder = ResNet(output_dim=2048, input_channels=3)

        # Replace the per-encoder fc with a shared fusion head
        # (ResNet.__init__ creates fc mapping 2048→output_dim; we override
        #  output_dim=2048 so fc is 2048→2048, then replace it with Identity)
        self.obs_encoder.fc = nn.Identity()
        self.goal_encoder.fc = nn.Identity()

        self.action_head = nn.Sequential(
            nn.Linear(2048 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, obs, goal, return_mask=False):
        """
        Args:
            obs:  (B, 3, H, W) — current SAPIEN observation (ImageNet-normalised)
            goal: (B, 3, H, W) — target video frame      (ImageNet-normalised)
        Returns:
            actions: (B, output_dim)
            masks:   (B, 2, H, W) or None
        """
        # --- Stage 1: dual attention masks ---
        concat = torch.cat([obs, goal], dim=1)            # (B, 6, H, W)
        raw_masks = self.mask_model(concat)                # (B, 2, H, W)
        masks = torch.sigmoid(raw_masks)
        masks_hard = torch.where(masks < 0.5, 0.0, 1.0)
        # STE: forward uses hard mask, backward uses soft mask
        masks_ste = (masks_hard - masks).detach() + masks  # (B, 2, H, W)

        obs_masked = obs * masks_ste[:, 0:1]
        goal_masked = goal * masks_ste[:, 1:2]

        # --- Stage 2: dual feature extraction ---
        feat_obs = self.obs_encoder(obs_masked)            # (B, 2048)
        feat_goal = self.goal_encoder(goal_masked)         # (B, 2048)

        # --- Stage 3: fusion + action prediction ---
        feat = torch.cat([feat_obs, feat_goal], dim=1)     # (B, 4096)
        actions = self.action_head(feat)                   # (B, output_dim)

        if return_mask:
            return actions, masks
        return actions, None

    @classmethod
    def from_pretrained_mask(cls, mask_state_dict, output_dim: int = 14):
        """Initialise from an existing single-input Mask checkpoint.

        Both ResNet encoders are initialised from the same pretrained
        ``resnet_model`` weights.  The UNet is initialised from the
        pretrained ``mask_model`` weights for the first 3 input channels;
        the remaining 3 channels (goal) are zero-initialised so the model
        initially behaves like the single-input IDM.
        """
        model = cls(output_dim=output_dim)

        # --- ResNet encoders (both from the same checkpoint) ---
        resnet_sd = {
            k.replace("resnet_model.", ""): v
            for k, v in mask_state_dict.items()
            if k.startswith("resnet_model.")
        }
        # The pretrained fc maps 2048→14; we replaced fc with Identity,
        # so skip fc weights.
        resnet_sd_no_fc = {k: v for k, v in resnet_sd.items() if not k.startswith("fc.")}
        model.obs_encoder.load_state_dict(resnet_sd_no_fc, strict=False)
        model.goal_encoder.load_state_dict(resnet_sd_no_fc, strict=False)

        # --- UNet (first 3 input channels from checkpoint) ---
        unet_sd = {
            k.replace("mask_model.", ""): v
            for k, v in mask_state_dict.items()
            if k.startswith("mask_model.")
        }
        # Manually handle the first conv (in_conv) which changes from 3→6 channels
        key_in_conv_w = "in_conv.0.weight"  # shape (mid_ch, 3, 3, 3)
        if key_in_conv_w in unet_sd:
            old_w = unet_sd[key_in_conv_w]           # (64, 3, 3, 3)
            new_w = model.mask_model.in_conv[0].weight  # (64, 6, 3, 3)
            new_w.data[:, :3] = old_w
            new_w.data[:, 3:] = 0.0                   # zero-init goal channels
            del unet_sd[key_in_conv_w]
        # The output conv changes from 1→2 channels
        key_out_w = "out.weight"                       # shape (1, base_ch, 1, 1)
        key_out_b = "out.bias"
        if key_out_w in unet_sd:
            old_w = unet_sd[key_out_w]                 # (1, 64, 1, 1)
            new_w = model.mask_model.out.weight         # (2, 64, 1, 1)
            new_w.data[0:1] = old_w                    # obs mask ← pretrained
            new_w.data[1:2] = old_w                    # goal mask ← same init
            del unet_sd[key_out_w]
        if key_out_b in unet_sd:
            old_b = unet_sd[key_out_b]                 # (1,)
            new_b = model.mask_model.out.bias           # (2,)
            new_b.data[0:1] = old_b
            new_b.data[1:2] = old_b
            del unet_sd[key_out_b]
        # Load remaining (shared) UNet weights
        model.mask_model.load_state_dict(unet_sd, strict=False)

        return model


class IDM(nn.Module):

    def __init__(self, model_name, *args, **kwargs):
        super(IDM, self).__init__()
        match model_name:
            case "mask":
                self.model = Mask(*args, **kwargs)
            case "goal_conditioned_mask":
                self.model = GoalConditionedMask(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")
        if self.model.output_dim == 14:
            train_mean = torch.tensor([-0.26866713, 0.83559588, 0.69520934, -0.29099351, 0.18849116, -0.01014598, 1.41953145, 0.35073715, 1.05651613, 0.8930193, -0.37493264, -0.18510782, -0.0272574, 1.35274259])
            train_std = torch.tensor([0.25945241, 0.65903812, 0.52147207, 0.42150272, 0.32029947, 0.28452226, 1.78270006, 0.29091741, 0.67675932, 0.58250554, 0.42399049, 0.28697442, 0.31100304, 1.67651926])
        else:
            train_mean = torch.tensor([-0.0011163579765707254, 0.3502498269081116, 0.010254094377160072, -2.0258395671844482, 0.06505978852510452, 2.3033766746520996, 0.8659588098526001, 0.026907790452241898, -0.027255306020379066])
            train_std = torch.tensor([0.12338999658823013, 0.35243555903434753, 0.17533640563488007, 0.43524453043937683, 0.416223406791687, 0.31947872042655945, 0.6888905167579651, 0.014177982695400715, 0.014080556109547615])
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, x):
        x = (x - self.train_mean) / self.train_std
        return x

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0] * self.train_std + self.train_mean, *output[1:]
        else:
            return output * self.train_std + self.train_mean


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim
        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(output_dim, 3)

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        else:
            return outputs, None
