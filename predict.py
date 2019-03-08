import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
import numpy as np
import time
import os
import subprocess

class Predict:

    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 6)
        self.model.load_state_dict(torch.load('model1.pt'))
        self.model.eval()
        self.tsr = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def inference(self, frame):
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        im1 = self.tsr(pil_im)
        im1 = im1.view(1,3,224,224)
        out = self.model(im1)
        out = out.detach().numpy()
        final = np.argmax(out)
        f = open("transfer.txt","w")
        f.write(str(final))
        f.close()
        subprocess.call("./scp.sh")



