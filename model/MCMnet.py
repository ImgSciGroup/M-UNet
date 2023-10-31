import torch.nn.functional as F

from model.MCMunet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(6, 64)
        self.mcm = MCM1(64, 64)
        # self.conmodel=ConvModule(64,64)
        self.down1 = Down(64, 128)
        self.mcm5 = MCM5(128,128)
        self.down2 = Down(128, 256)
        self.mcm2 = MCM2(256, 256)
        self.down3 = Down(256, 512)
        self.mcm3=MCM3(512,512)
        self.cae=CAENet()
        self.cae1=CAENet1()
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x1,x2):
        x = torch.cat((x1, x2), 1)

        x1 = self.inc(x)
        x1 = self.mcm(x1)
        x2 = self.down1(x1)
        x2=self.mcm5(x2)

        x3 = self.down2(x2)
        x3 = self.mcm2(x3)

        x4 = self.down3(x3)
        x4=self.mcm3(x4)

        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)

        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
if __name__ == '__main__':
    in1=torch.randn(1,3,64,64)
    in2=torch.randn(1,3,64,64)
    net=UNet(3,1)
    out=net(in1,in2)
    print(out.shape)