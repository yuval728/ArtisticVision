import torch
import utils
import transformer
import os
import cv2
from stylize import stylize
import argparse
import time

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames, info

def save_video(frames, info, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For MP4
    out = cv2.VideoWriter(output_path, fourcc, info['fps'], (info['width'], info['height']))
    for frame in frames:
        out.write(frame)
    out.release()
    
def load_transformer(args, device):
    transformer_net = transformer.TransformerNetwork()
    transformer_net.to(device)
    transformer_net.load_state_dict(torch.load(args.model, map_location=device ,weights_only=True))
    return transformer_net

def stylize_video(args):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    start = time.time()
    frames, info = load_video(args.video)
    transformer_net = load_transformer(args, device)
    with torch.inference_mode():
        stylized_frames = []
        for frame in frames:
            content_tensor = utils.image_to_tensor(frame).to(device)
            generated_tensor = transformer_net(content_tensor)
            generated_image = utils.tensor_to_image(generated_tensor.detach())
            if args.preserve_color:
                generated_image = utils.transfer_color(frame, generated_image)
            stylized_frames.append(generated_image)
    print('Stylized {} frames in {:.2f}s'.format(info['total_frames'], time.time() - start))
    save_video(stylized_frames, info, args.output_video)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_video', type=str, required=True)
    parser.add_argument('--preserve_color', action='store_true')
    args = parser.parse_args()
    stylize_video(args)
    
    