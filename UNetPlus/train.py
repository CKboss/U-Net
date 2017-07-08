import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    from cv2 import cv2
except:
    import cv2

from UNetPlus.UNetPlus import UNetPlus
from UNetPlus.image_gen import RgbDataProvider

class UNet(nn.Module):

    def __init__(self):
        super(UNet,self).__init__()
        self.unet = UNetPlus().cuda()

    def forward(self,x):
        _,_,h,w = x.size()
        x = self.unet(x)
        return x

lossfunc = nn.CrossEntropyLoss()

model = UNet().cuda()

optim = torch.optim.Adagrad(model.parameters(),lr=0.001)

dsl = RgbDataProvider(256,256)


for i in range(5000):

    img,label = dsl.next_data()
    Img = np.array(img)
    Label = np.array(label)
    # label = label.reshape(-1,1)

    img = Variable(torch.from_numpy(img)).cuda().float()
    img = torch.unsqueeze(img,0)
    label = Variable(torch.from_numpy(label)).cuda().long()
    label = torch.unsqueeze(label,0)

    model.zero_grad()

    out = model(img)

    loss = lossfunc(out,label)

    loss.backward()

    if i%200 == 0:
        pi = F.softmax(out)
        ppi = np.argmax(pi.cpu().data.numpy(),1).reshape((256,256))
        plt.subplot(131)
        plt.imshow(Img[0])
        plt.subplot(132)
        plt.imshow(ppi)
        plt.subplot(133)
        plt.imshow(Label)

        plt.show()


    print('i: {} loss: {}'.format(i,loss.data[0]))

    optim.step()

