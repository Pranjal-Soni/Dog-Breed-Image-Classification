import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    

class DogBreedClassification(ImageClassificationBase):
    def __init__(self,num_classes):
        super().__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,num_classes)
    
    def forward(self, xb):
        return self.resnet(xb)