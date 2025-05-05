import torch
import torchvision
from torchvision.models import ResNet101_Weights, MaxVit_T_Weights, ViT_B_16_Weights, Swin_B_Weights

def resnet101_backbone(freeze_backbone: bool = False, num_classes=4):
    model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def vit_backbone(freeze_backbone: bool = False, num_classes=4):
    model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.head.parameters():
            param.requires_grad = True

    return model


def swin_backbone(freeze_backbone: bool = False, num_classes=4):
    model = torchvision.models.swin_b(weights=Swin_B_Weights.DEFAULT)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    return model

def maxvit_backbone(freeze_backbone: bool = False, num_classes=4):
    model = torchvision.models.maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[5].in_features
    model.classifier[5] = torch.nn.Linear(in_features, num_classes, bias=False)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
