import cv2
import torch
import numpy as np

def preprocess(img):
    img = cv2.resize(img, (256,256))
    img = img/255.0
    # img = img.permute(2,0,1)
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    return img

def postprocess(tensor):
    img = tensor.squeeze().permute(1,2,0).detach().numpy()
    img = np.clip(img*255,0,255).astype(np.uint8)
    
    return img