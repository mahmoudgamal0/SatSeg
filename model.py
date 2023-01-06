import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """

        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

# upsample tensor 'src' to have the same spatial size with tensor 'tar'


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], align_corners=True, mode='bilinear')
    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.att6 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att5 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att4 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att3 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att2 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att1 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hatt6 = self.att6(gate=hx7, skip_connection=hx6)
        hx6d = self.rebnconv6d(torch.cat((hatt6, hx7), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hatt5 = self.att5(gate=hx6dup, skip_connection=hx5)
        hx5d = self.rebnconv5d(torch.cat((hatt5, hx6dup), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hatt4 = self.att4(gate=hx5dup, skip_connection=hx4)
        hx4d = self.rebnconv4d(torch.cat((hatt4, hx5dup), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hatt3 = self.att3(gate=hx4dup, skip_connection=hx3)
        hx3d = self.rebnconv3d(torch.cat((hatt3, hx4dup), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hatt2 = self.att2(gate=hx3dup, skip_connection=hx2)
        hx2d = self.rebnconv2d(torch.cat((hatt2, hx3dup), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hatt1 = self.att1(gate=hx2dup, skip_connection=hx1)
        hx1d = self.rebnconv1d(torch.cat((hatt1, hx2dup), 1))

        return hx1d + hxin

### RSU-6 ###


class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.att5 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att4 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att3 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att2 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att1 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hatt5 = self.att5(gate=hx6, skip_connection=hx5)
        hx5d = self.rebnconv5d(torch.cat((hatt5, hx6), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hatt4 = self.att4(gate=hx5dup, skip_connection=hx4)
        hx4d = self.rebnconv4d(torch.cat((hatt4, hx5dup), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hatt3 = self.att3(gate=hx4dup, skip_connection=hx3)
        hx3d = self.rebnconv3d(torch.cat((hatt3, hx4dup), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hatt2 = self.att2(gate=hx3dup, skip_connection=hx2)
        hx2d = self.rebnconv2d(torch.cat((hatt2, hx3dup), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hatt1 = self.att1(gate=hx2dup, skip_connection=hx1)
        hx1d = self.rebnconv1d(torch.cat((hatt1, hx2dup), 1))

        return hx1d + hxin

### RSU-5 ###


class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.att4 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att3 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att2 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att1 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hatt4 = self.att4(gate=hx5, skip_connection=hx4)
        hx4d = self.rebnconv4d(torch.cat((hatt4, hx5), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hatt3 = self.att3(gate=hx4dup, skip_connection=hx3)
        hx3d = self.rebnconv3d(torch.cat((hatt3, hx4dup), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hatt2 = self.att2(gate=hx3dup, skip_connection=hx2)
        hx2d = self.rebnconv2d(torch.cat((hatt2, hx3dup), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hatt1 = self.att1(gate=hx2dup, skip_connection=hx1)
        hx1d = self.rebnconv1d(torch.cat((hatt1, hx2dup), 1))

        return hx1d + hxin

### RSU-4 ###


class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.att3 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att2 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.att1 = AttentionBlock(
            F_g=mid_ch, F_l=mid_ch, n_coefficients=mid_ch//2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hatt3 = self.att3(gate=hx4, skip_connection=hx3)
        hx3d = self.rebnconv3d(torch.cat((hatt3, hx4), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hatt2 = self.att2(gate=hx3dup, skip_connection=hx2)
        hx2d = self.rebnconv2d(torch.cat((hatt2, hx3dup), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hatt1 = self.att1(gate=hx2dup, skip_connection=hx1)
        hx1d = self.rebnconv1d(torch.cat((hatt1, hx2dup), 1))

        return hx1d + hxin

### RSU-4F ###


class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


def _transform_output(d, training=False):
    vect = d
    # if training:
        # vect = F.log_softmax(vect, 1)
    # else:
    #     import ipdb; ipdb.set_trace()
    #     vect = (torch.argmax(vect, 1) + 1).type(torch.IntTensor).to("cuda:0")
    return vect

##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self, x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        # Softmax
        return _transform_output(d0, self.training), _transform_output(d1, self.training), _transform_output(d2, self.training), _transform_output(d3, self.training), _transform_output(d4, self.training), _transform_output(d5, self.training), _transform_output(d6, self.training)


if __name__ == "__main__":
  model = U2NET().cuda()
  summary(model, (3, 64, 64))