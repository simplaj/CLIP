'''
An image search app based on CLIP
author: tzh
'''
import os
from PIL import Image
import clip
import torch


def load_image(dir_path, preprocess, device):
    names = os.listdir(dir_path)
    paths = [os.path.join(dir_path, x) for x in names]
    ims = []
    for im_path in paths:
        im = Image.open(im_path)
        im = preprocess(im)
        ims.append(im)
    ims = torch.stack(ims, dim=0)
    return ims, names


def cal_im_f(im, model):
    '''
    Calculate features
    
    '''
    with torch.no_grad():
        image_features = model.encode_image(im)
    return image_features


def cal_text_f(text, model):
    '''
    Calculate features
    
    '''
    with torch.no_grad():
       text_features = model.encode_text(text)
    return text_features


def pick_topk(image_features, text_features):
    '''
    Pick the top 5 most similar labels for the image
    
    '''
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    return values, indices


def print_res(values, indices, im_names):
    '''
    Print the result
    '''
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{im_names[index]}: {100 * value.item():.2f}%")
        

def process():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    
    dir_name = './images'
    im_t, im_names = load_image(dir_name, preprocess, device)
    im_f = cal_im_f(im_t, model)
    while True:
        text = input('input the query:')
        text = clip.tokenize(text).to(device)
        text_f = cal_text_f(text, model)
        v, idx = pick_topk(im_f, text_f)
        print_res(v, idx, im_names)


if __name__ == '__main__':
    process()