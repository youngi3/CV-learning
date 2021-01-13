import torch
import torch.nn as nn
import torchvision.models as models

def model_modify():
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier[1] = nn.Sequential(
                                            nn.Linear(1280, 640),
                                            nn.ReLU6(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(640,62)
    )
    print(mobilenet)
    return mobilenet


