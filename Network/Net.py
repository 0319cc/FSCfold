import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SkipConv(nn.Module):
    def __init__(self, ch_in, CatChannels, kernel_size=0, stride=0, scale_factor=0, concat=False, upsample=False):
        super(SkipConv, self).__init__()
        layers = []
        if not concat:
            if not upsample:
                layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
            else:
                layers.append(nn.Upsample(scale_factor=scale_factor))
                # layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(ch_in, CatChannels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(CatChannels))
        layers.append(nn.ReLU(inplace=True))
        self.skip = nn.Sequential(*layers)

    def forward(self, x):
        return self.skip(x)

class FSCfold(nn.Module):
    def __init__(self, img_ch=17, output_ch=1):
        super(FSCfold, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Encoder
        self.conv1 = ConvBlock(img_ch, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.conv5 = ConvBlock(filters[3], filters[4])

        ## Decoder
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.h1_PT_hd4 = SkipConv(filters[0], self.CatChannels, kernel_size=8, stride=8)
        self.h2_PT_hd4 = SkipConv(filters[1], self.CatChannels, kernel_size=4, stride=4)
        self.h3_PT_hd4 = SkipConv(filters[2], self.CatChannels, kernel_size=2, stride=2)
        self.h4_Cat_hd4 = SkipConv(filters[3], self.CatChannels, concat=True)
        self.hd5_UT_hd4 = SkipConv(filters[4], self.CatChannels, scale_factor=2, upsample=True)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=True)
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.h1_PT_hd3 = SkipConv(filters[0], self.CatChannels, kernel_size=4, stride=4)
        self.h2_PT_hd3 = SkipConv(filters[1], self.CatChannels, kernel_size=2, stride=2)
        self.h3_Cat_hd3 = SkipConv(filters[2], self.CatChannels, concat=True)
        self.hd4_UT_hd3 = SkipConv(self.UpChannels, self.CatChannels, scale_factor=2, upsample=True)
        self.hd5_UT_hd3 = SkipConv(filters[4], self.CatChannels, scale_factor=4, upsample=True)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=True)
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.h1_PT_hd2 = SkipConv(filters[0], self.CatChannels, kernel_size=2, stride=2)
        self.h2_Cat_hd2 = SkipConv(filters[1], self.CatChannels, concat=True)
        self.hd3_UT_hd2 = SkipConv(self.UpChannels, self.CatChannels, scale_factor=2, upsample=True)
        self.hd4_UT_hd2 = SkipConv(self.UpChannels, self.CatChannels, scale_factor=4, upsample=True)
        self.hd5_UT_hd2 = SkipConv(filters[4], self.CatChannels, scale_factor=8, upsample=True)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=True)
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.h1_Cat_hd1 = SkipConv(filters[0], self.CatChannels, concat=True)
        self.hd2_UT_hd1 = SkipConv(self.UpChannels, self.CatChannels, scale_factor=2, upsample=True)
        self.hd3_UT_hd1 = SkipConv(self.UpChannels, self.CatChannels, scale_factor=4, upsample=True)
        self.hd4_UT_hd1 = SkipConv(self.UpChannels, self.CatChannels, scale_factor=8, upsample=True)
        self.hd5_UT_hd1 = SkipConv(filters[4], self.CatChannels, scale_factor=16, upsample=True)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, bias=True)
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.Conv_1x1 = nn.Conv2d(self.UpChannels, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        ## Encoder
        h1 = self.conv1(inputs)  # h1->320*320*64
        h2 = self.conv2(self.Maxpool(h1))  # h2->160*160*128
        h3 = self.conv3(self.Maxpool(h2))  # h3->80*80*256
        h4 = self.conv4(self.Maxpool(h3))  # h4->40*40*512
        hd5 = self.conv5(self.Maxpool(h4))  # h5->20*20*1024

        ## Decoder
        h1_PT_hd4 = self.h1_PT_hd4(h1)
        h2_PT_hd4 = self.h2_PT_hd4(h2)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4(h4)
        hd5_UT_hd4 = self.hd5_UT_hd4(hd5)
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3(h1)
        h2_PT_hd3 = self.h2_PT_hd3(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3(hd4)
        hd5_UT_hd3 = self.hd5_UT_hd3(hd5)
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2(hd3)
        hd4_UT_hd2 = self.hd4_UT_hd2(hd4)
        hd5_UT_hd2 = self.hd5_UT_hd2(hd5)
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1(hd2)
        hd3_UT_hd1 = self.hd3_UT_hd1(hd3)
        hd4_UT_hd1 = self.hd4_UT_hd1(hd4)
        hd5_UT_hd1 = self.hd5_UT_hd1(hd5)
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d1 = self.Conv_1x1(hd1)  # d1->320*320*output_ch
        d1 = d1.squeeze(1)
        return torch.transpose(d1, -1, -2) * d1



