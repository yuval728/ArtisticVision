import torch
import utils
import transformer
import os
from torchvision import transforms
import time
import cv2
import argparse

def stylize(args):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device  = torch.device('cpu')
    # Load transformer network
    transformer_net = transformer.TransformerNetwork()
    transformer_net.to(device)
    transformer_net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    
    with torch.inference_mode():
        
        torch.cuda.empty_cache()
        content_image = utils.load_image(args.content_image)
        content_tensor = utils.image_to_tensor(content_image).to(device)
        generated_tensor = transformer_net(content_tensor)
        generated_image = utils.tensor_to_image(generated_tensor.detach())
        if args.preserve_color:
            generated_image = utils.transfer_color(content_image, generated_image)
        utils.show_image(generated_image)
        utils.save_image(generated_image, args.output_image)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_image', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_image', type=str, required=True)
    parser.add_argument('--preserve_color', action='store_true')
    args = parser.parse_args()
    stylize(args)