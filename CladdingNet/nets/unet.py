import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import DenseVGG16

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_size, in_size, kernel_size=3, padding=1, groups=in_size)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_size, out_size, kernel_size=1)
        # 额外的卷积层，用于更好地融合特征
        self.conv1 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # 上采样 inputs2，使其与 inputs1 具有相同的空间尺寸
        upsampled_inputs2 = F.interpolate(inputs2, scale_factor=2, mode='bilinear', align_corners=True)
        # 深度可分离卷积
        outputs = self.depthwise(upsampled_inputs2)
        outputs = self.pointwise(outputs)
        # 拼接上采样后的特征和输入特征
        outputs = torch.cat([inputs1, outputs], 1)
        # 两层卷积进行进一步处理
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = DenseVGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
