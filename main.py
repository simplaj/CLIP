import os
from PIL import Image
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


def load_image(dir_path, preprocess, device):
    names = os.listdir(dir_path)
    paths = [os.path.join(dir_path, x) for x in names]
    ims = []
    for im_path in paths:
        im = Image.open(im_Path)
        im = preprocess(im)
        ims.append(im)
    ims = torch.stack(ims, dim=0)
    print(ims.shape)

# Download the dataset

# Prepare the inputs
def pre_inputs():
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
def cal_f():
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
def pick_topk():
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

# Print the result
def print_res():
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
        
        
if __name__ == '__main__':
    dir_name = './images'
    load_image(dir_name, preprocess, device)