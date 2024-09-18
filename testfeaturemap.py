import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
resnet34 = models.resnet34(pretrained=True)
modules = list(resnet34.children())[:-1]
modules2 = list(resnet34.children())[:-2]
modules3 = list(resnet34.children())[:-3]
modules4 = list(resnet34.children())[:-4]
print(modules)
print(modules2)
print(modules3)
print(modules4)
resnet34 = nn.Sequential(*modules3)
for p in resnet34.parameters():
    p.requires_grad = False
x = torch.rand((1, 3, 3584, 3584)).cuda()
resnet34 = resnet34.cuda()
feature = resnet34(x)
print(feature)
