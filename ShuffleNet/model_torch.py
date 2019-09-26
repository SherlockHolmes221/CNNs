import torch.nn as nn


class ShuffleBlock(nn.Module):
    def __init__(self, group):
        super(ShuffleBlock, self).__init__()
        self.group = group

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.group
        return x.view(N, g, int(C/g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = int(out_planes / 4)

        g = 1 if in_planes == 24 else groups

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.shuffle1 = ShuffleBlock(group=g)

        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                               groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)





class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet, self).__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    ShuffleNet()
