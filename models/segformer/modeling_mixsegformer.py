import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

class MoEBase(nn.Module):
    def __init__(self):
        super(MoEBase, self).__init__()
        self.scores = None
        self.router = None

    def set_score(self, scores):
        self.scores = scores
        for module in self.modules():
            if hasattr(module, 'scores'):
                module.scores = self.scores

class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, scores):  # binarization

        expert_pred = torch.argmax(scores, dim=1)  # [bs]
        expert_pred_one_hot = torch.zeros_like(scores).scatter_(1, expert_pred.unsqueeze(-1), 1)

        return expert_pred, expert_pred_one_hot

    @staticmethod
    def backward(ctx, g1, g2):
        return g2


def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"

class MoEConv(nn.Conv2d, MoEBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 n_expert=5):
        super(MoEConv, self).__init__(in_channels, out_channels * n_expert, kernel_size, stride, padding, dilation,
            groups, bias, )
        self.in_channels = in_channels
        self.out_channels = out_channels * n_expert
        self.expert_width = out_channels

        self.n_expert = n_expert
        assert self.n_expert >= 1
        self.layer_selection = torch.zeros([n_expert, self.out_channels])
        for cluster_id in range(n_expert):
            start = cluster_id * self.expert_width
            end = (cluster_id + 1) * self.expert_width
            idx = torch.arange(start, end)
            self.layer_selection[cluster_id][idx] = 1
        self.scores = None

    def forward(self, x):
        if self.n_expert > 1:
            if self.scores is None:
                self.scores = self.router(x)
            expert_selection, expert_selection_one_hot = GetMask.apply(self.scores)
            mask = torch.matmul(expert_selection_one_hot, self.layer_selection.to(x))  # [bs, self.out_channels]
            out = super(MoEConv, self).forward(x)
            out = out * mask.unsqueeze(-1).unsqueeze(-1)
            index = torch.where(mask.view(-1) > 0)[0]
            shape = out.shape
            out_selected = out.view(shape[0] * shape[1], shape[2], shape[3])[index].view(shape[0], -1, shape[2],
                                                                                         shape[3])
        else:
            out_selected = super(MoEConv, self).forward(x)
        self.scores = None
        return out_selected
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(Block, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            diff = planes - in_planes
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, int(diff * 0.5), int((diff + 1) * 0.5)), "constant", 0))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
class Router(nn.Module):
    def __init__(self, block, num_blocks, num_experts=2):
        super(Router, self).__init__()
        self.in_planes = 16
        self.conv_layer = nn.Conv2d

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_experts)

    def _make_layer(self, block, planes, num_blocks, stride):
        planes = planes
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def build_router(**kwargs):
    return Router(Block, [3, 3, 3], **kwargs)
    
if __name__ == "__main__": 
    images = torch.randn(1, 3, 224, 224)
    model = MoEConv(3, 64, 3, n_expert=5)
    router = build_router()
    model.router = router
    out = model(images)
    print(out.shape)
    print("Done!")