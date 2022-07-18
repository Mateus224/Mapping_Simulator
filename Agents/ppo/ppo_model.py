import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet(nn.Module):
    def __init__(self,device, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
        super().__init__()
        self.device = device
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [128, 128, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (
                    i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))

        #self.gap = torch.nn.AdaptiveAvgPool2d(1)
        #self.fc = torch.nn.Linear(filters[4], outputs)
        self.fc_h_v01 = nn.Linear(4608, 512)
        self.fc_h_v02 = nn.Linear(512, 512)
        self.fc_h_v03 = nn.Linear(512, 512)
        self.fc_h_a01 = nn.Linear(4608, 512)
        self.fc_h_a02 = nn.Linear(512, 512)
        self.fc_h_a03 = nn.Linear(512, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512, 8)

    def forward(self, input):
        input=torch.Tensor(input).to(self.device)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = torch.flatten(input, start_dim=1)

        value=self.fc_z_v(F.relu((self.fc_h_v02(F.relu(self.fc_h_v01(input))))))
        policy = F.softmax(self.fc_z_a(F.relu(self.fc_h_a02(F.relu(self.fc_h_a01(input))))), dim=-1)

        #value = self.fc_z_v(F.relu(input))
        #police= F.softmax(self.fc_z_a(F.relu(input)))
        #input = self.gap(input)
        ## torch.flatten()
        ## https://stackoverflow.com/questions/60115633/pytorch-flatten-doesnt-maintain-batch-size
        #input = torch.flatten(input, start_dim=1)
        #input = self.fc(input)

        return  policy, value

#resnet18 = ResNet(4, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=128)
#resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet18, (4, 27, 27))
#resnet34 = ResNet(3, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=1000)
#resnet34.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet34, (3, 224, 224))
#resnet50 = ResNet(3, ResBottleneckBlock, [
#                  3, 4, 6, 3], useBottleneck=True, outputs=1000)
#resnet50.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet50, (3, 224, 224))
#resnet101 = ResNet(3, ResBottleneckBlock, [
#                   3, 4, 23, 3], useBottleneck=True, outputs=1000)
#resnet101.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet101, (3, 224, 224))
#resnet152 = ResNet(3, ResBottleneckBlock, [
#                   3, 8, 36, 3], useBottleneck=True, outputs=1000)
#resnet152.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet152, (3, 224, 224))
