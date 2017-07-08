import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UNetPlus(nn.Module):

    def __init__(self,klassnum=2):

        super(UNetPlus,self).__init__()

        self.klassnum = klassnum

        self.conv_1_1 = nn.Conv2d(3,64,3,padding=1)
        self.conv_1_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.mxpool_1 = nn.MaxPool2d(2)

        self.conv_2_1 = nn.Conv2d(64,128,3,padding=1)
        self.conv_2_2 = nn.Conv2d(128,128,3,padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.mxpool_2 = nn.MaxPool2d(2)

        self.conv_3_1 = nn.Conv2d(128,256,3,padding=1)
        self.conv_3_2 = nn.Conv2d(256,256,3,padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.mxpool_3 = nn.MaxPool2d(2)

        self.conv_4_1 = nn.Conv2d(256,512,3,padding=1)
        self.conv_4_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.mxpool_4 = nn.MaxPool2d(2)

        self.conv_5_1 = nn.Conv2d(512,1024,3,padding=1)
        self.conv_5_2 = nn.Conv2d(1024,1024,3,padding=1)
        self.conv_5_up = nn.Conv2d(1024,512,3,padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn5_out = nn.BatchNorm2d(512)

        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_6_1 = nn.Conv2d(1024,512,3,padding=1)
        self.conv_6_2 = nn.Conv2d(512,512,3,padding=1)
        self.conv_6_up = nn.Conv2d(512,256,3,padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn6_out = nn.BatchNorm2d(256)

        self.conv_7_1 = nn.Conv2d(512,256,3,padding=1)
        self.conv_7_2 = nn.Conv2d(256,256,3,padding=1)
        self.conv_7_up = nn.Conv2d(256,128,3,padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn7_out = nn.BatchNorm2d(128)


        self.conv_8_1 = nn.Conv2d(256,128,3,padding=1)
        self.conv_8_2 = nn.Conv2d(128,128,3,padding=1)
        self.conv_8_up = nn.Conv2d(128,64,3,padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn8_out = nn.BatchNorm2d(64)

        self.conv_9_1 = nn.Conv2d(128,64,3,padding=1)
        self.conv_9_2 = nn.Conv2d(64,64,3,padding=1)
        self.conv_9_3 = nn.Conv2d(64,self.klassnum,1)
        self.bn9 = nn.BatchNorm2d(self.klassnum)

        self._initialize_weights()

    def forward(self,x):

        x = self.conv_1_1(x)
        x = F.relu(x)
        x_1 = self.conv_1_2(x)
        x = self.bn1(x_1)
        x = F.relu(x)

        x = self.mxpool_1(x_1)

        x = self.conv_2_1(x)
        x = F.relu(x)
        x_2 = self.conv_2_2(x)
        x = self.bn2(x_2)
        x = F.relu(x)

        x = self.mxpool_2(x_2)

        x = self.conv_3_1(x)
        x = F.relu(x)
        x_3 = self.conv_3_2(x)
        x = self.bn3(x_3)
        x = F.relu(x)

        x = self.mxpool_3(x_3)

        x = self.conv_4_1(x)
        x = F.relu(x)
        x_4 = self.conv_4_2(x)
        x = self.bn4(x_4)
        x = F.relu(x)

        x = self.mxpool_4(x_4)

        x = self.conv_5_1(x)
        x = F.relu(x)
        x = self.conv_5_2(x)
        x = self.bn5(x)

        x = self.Upsample(x)
        x = self.conv_5_up(x)
        x = self.bn5_out(x)
        x = torch.cat([x,x_4],1)

        x = self.conv_6_1(x)
        x = F.relu(x)
        x = self.conv_6_2(x)
        x = self.bn6(x)

        x = self.Upsample(x)
        x = self.conv_6_up(x)
        x = self.bn6_out(x)
        x = torch.cat([x,x_3],1)

        x = self.conv_7_1(x)
        x = F.relu(x)
        x = self.conv_7_2(x)
        x = self.bn7(x)

        x = self.Upsample(x)
        x = self.conv_7_up(x)
        x = self.bn7_out(x)
        x = torch.cat([x,x_2],1)

        x = self.conv_8_1(x)
        x = F.relu(x)
        x = self.conv_8_2(x)
        x = self.bn8(x)

        x = self.Upsample(x)
        x = self.conv_8_up(x)
        x = self.bn8_out(x)
        x = torch.cat([x,x_1],1)

        x = self.conv_9_1(x)
        x = F.relu(x)
        x = self.conv_9_2(x)
        x = F.relu(x)
        x = self.conv_9_3(x)
        x = F.relu(x)

        x = self.bn9(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    unet = UNetPlus().cuda()
    x = torch.randn((1,1,256,256))
    x = Variable(x).cuda()
    y = unet(x)
    import IPython; IPython.embed()