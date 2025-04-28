
import cv2
import torch
import os
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def load_midas_model(size):

    d_model_type = {"large": "DPT_Large", "hybrid":"DPT_Hybrid","small":"MiDaS_small"}
    model_type = d_model_type[size]

    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return transform, midas




def get_depth(img_path,transform,midas):
    #0: white, 1000?:black
    img = cv2.imread(img_path)
    #plt.imshow(img)
    #plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        #Interpolate: Upsamples or downsamples the tensor to match the target size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    #plt.imshow(output,cmap='gray')
    #plt.show()
    return output





