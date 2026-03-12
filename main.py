import os
os.environ['KMP_DUPLICATE_LIB_OK']="True"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import enhance_net_nopool
import Myloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load pretrained model
# ----------------------------

def load_model(weights):

    model = enhance_net_nopool().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    return model


# ----------------------------
# Image Loader
# ----------------------------

def load_image(path):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype('float32') / 255.0

    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

    return tensor.to(device), img

def load_mobile_image(path):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (256,256))

    img = img.astype('float32') / 255.0

    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

    return tensor.to(device), img


# ----------------------------
# Enhancement
# ----------------------------

def enhance_image(model, img_tensor):

    with torch.no_grad():

        _,enhanced,_ = model(img_tensor)

    enhanced = enhanced.squeeze().permute(1,2,0).cpu().numpy()
   

    enhanced = np.clip(enhanced,0,1)

    return enhanced


# ----------------------------
# Fine Tune Per Image
# ----------------------------

def finetune_image(model, img_tensor, iterations=300):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_spa = Myloss.L_spa()
    loss_exp = Myloss.L_exp(16,0.6)
    loss_col = Myloss.L_color()
    loss_tv = Myloss.L_TV()

    for _ in range(iterations):

        _,enhanced,b = model(img_tensor)

        loss = (
            loss_spa(enhanced, img_tensor)
            + loss_exp(enhanced)
            + loss_col(enhanced)
            + 200 * loss_tv(b)
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    enhanced = enhanced.detach().cpu().squeeze().permute(1,2,0).numpy()

    return np.clip(enhanced,0,1)


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(pred, gt):

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1)

    return psnr, ssim


# ----------------------------
# Visualization
# ----------------------------

def save_visuals(low, pre, fine, gt, name):

    fig,ax = plt.subplots(1,4,figsize=(16,5))

    ax[0].imshow(low)
    ax[0].set_title("Low Light")

    ax[1].imshow(pre)
    ax[1].set_title("Pretrained")

    ax[2].imshow(fine)
    ax[2].set_title("Finetuned")

    if gt is not None:
        ax[3].imshow(gt)
        ax[3].set_title("Ground Truth")

    for a in ax:
        a.axis("off")

    plt.savefig(f"results/{name}.png")
    plt.close()


# ----------------------------
# Main Experiment
# ----------------------------

def run_experiment():

    os.makedirs("results", exist_ok=True)

    model = load_model("snapshots/Epoch99.pth")

    low_dir = "data/VE-LOL-L-Syn/VE-LOL-L-Syn-Low_train"
    high_dir = "data/VE-LOL-L-Syn/VE-LOL-L-Syn-Normal_train"

    rows = []

    for name in tqdm(os.listdir(low_dir)[:5]):

        low_path = os.path.join(low_dir,name)
        high_path = os.path.join(high_dir,name)

        img_tensor, low_img = load_image(low_path)

        gt = cv2.imread(high_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = gt.astype('float32')/255.0

        # pretrained
        pre = enhance_image(model,img_tensor)

        psnr_pre, ssim_pre = compute_metrics(pre,gt)

        # finetune
        fine = finetune_image(model,img_tensor)

        psnr_fine, ssim_fine = compute_metrics(fine,gt)

        rows.append([
            name,
            psnr_pre,
            ssim_pre,
            psnr_fine,
            ssim_fine
        ])

        save_visuals(low_img,pre,fine,gt,name)

    df = pd.DataFrame(rows,columns=[
        "Image",
        "PSNR Pretrained",
        "SSIM Pretrained",
        "PSNR Finetuned",
        "SSIM Finetuned"
    ])

    df.to_csv("results/results.csv",index=False)


    smartphone_dir = 'smartphone/'

    for name in tqdm(os.listdir(smartphone_dir)):

        low_path = os.path.join(smartphone_dir,name)

        img_tensor, low_img = load_mobile_image(low_path)

        # pretrained
        pre = enhance_image(model,img_tensor)

        # finetune
        fine = finetune_image(model,img_tensor)

        save_visuals(low_img,pre,fine,None,name)


if __name__ == "__main__":
    run_experiment()