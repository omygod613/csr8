import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)#8        

        return x


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model_ft = models.resnet34(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)#8        

        return x


class ResNet34_512(nn.Module):
    def __init__(self):
        super(ResNet34_512, self).__init__()
        self.model_ft = models.resnet34(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.transition = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),)
        self.globalPool = nn.Sequential(nn.MaxPool2d(16))
        self.prediction = nn.Sequential(nn.Linear(512, 8), nn.Sigmoid())

    def forward(self, x):
        # x = self.model_ft(x)
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)

        x = self.transition(x)
        x = self.globalPool(x)
        x = x.view(x.size(0), -1)
        x = self.prediction(x)#8        

        return x


class VGG11BN(nn.Module):
    def __init__(self):
        super(VGG11BN, self).__init__()
        self.model_ft = models.vgg11_bn(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class VGG19BN(nn.Module):
    def __init__(self):
        super(VGG19BN, self).__init__()
        self.model_ft = models.vgg19_bn(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model_ft = models.alexnet(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class DenseNet169(nn.Module):
    def __init__(self):
        super(DenseNet169, self).__init__()
        self.model_ft = models.densenet169(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self):
        super(DenseNet201, self).__init__()
        self.model_ft = models.densenet201(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class SqueezeNet1_1(nn.Module):
    def __init__(self):
        super(SqueezeNet1_1, self).__init__()
        self.model_ft = models.squeezenet1_1(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x


class SqueezeNet1_0(nn.Module):
    def __init__(self):
        super(SqueezeNet1_0, self).__init__()
        self.model_ft = models.squeezenet1_0(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x
        

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model_ft = models.inception_v3(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        self.prediction = nn.Sequential(nn.Linear(1000, 8), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        x = self.prediction(x)
        return x
