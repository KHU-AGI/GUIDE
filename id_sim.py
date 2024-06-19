import torch
import os

import torch.nn as nn

from arcface import IR_101
from mtcnn.mtcnn import MTCNN
from torchvision import transforms
from PIL import Image

class IDSimNet(nn.Module):
    def __init__(self):
        super(IDSimNet, self).__init__()
        ckpt_path = "CurricularFace_Backbone.pth"
        self.facenet = IR_101(input_size=112)
        self.facenet.load_state_dict(torch.load(ckpt_path))
        self.facenet.eval()
        self.mtcnn = MTCNN()

        self.transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def forward(self, img1, img2):
        assert (isinstance(img1, Image.Image) or isinstance(img1, str)) and (isinstance(img2, Image.Image) or isinstance(img2, str)), "img1 and img2 must be PIL.Image.Image or str"

        if isinstance(img1, str):
            img1 = Image.open(img1)
        img1, _ = self.mtcnn.align(img1)
        id1 = self.facenet(self.transform(img1).unsqueeze(0).cuda())[0]
        
        if isinstance(img2, str):
            img2 = Image.open(img2)
        img2, _ = self.mtcnn.align(img2)
        id2 = self.facenet(self.transform(img2).unsqueeze(0).cuda())[0]

        return id1.dot(id2).float()

if __name__ == "__main__":
    device = torch.device("cuda")
    id_sim_fn = IDSimNet().to(device)
    img1 = os.path.join("fake_images", "pretrained", "0", "0.png")
    img2 = os.path.join("fake_images", "pretrained", "0", "1.png")
    print(id_sim_fn(img1, img2).item())