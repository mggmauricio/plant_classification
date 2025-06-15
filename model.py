# model.py
import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    """Cria o modelo MobileNetV2 com Transfer Learning."""

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model
