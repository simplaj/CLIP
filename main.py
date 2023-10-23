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
        im = Image.open(im_path)
        im = preprocess(im)
        ims.append(im)
    ims = torch.stack(ims, dim=0)
    return ims


def pre_inputs():
    '''
    Prepare the inputs
    '''
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)


def cal_im_f(im, model):
    '''
    Calculate features
    
    '''
    with torch.no_grad():
        image_features = model.encode_image(im)
    return image_features


def cal_text_f(im, model):
    '''
    Calculate features
    
    '''
    with torch.no_grad():
        image_features = model.encode_image(image_input)


def pick_topk():
    '''
    Pick the top 5 most similar labels for the image
    
    '''
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)


def print_res():
    '''
    Print the result
    '''
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
        
        
if __name__ == '__main__':
    dir_name = './images'
    im_t = load_image(dir_name, preprocess, device)
    im_f = cal_im_f(im_t, model)
    print(im_f.shape)