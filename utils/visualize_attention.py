import os
import sys
import json
import yaml
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import PIL

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from ml_collections import ConfigDict
from tqdm import tqdm

sys.path.append(".") # to enable execution from parent directory
sys.path.append("..") # to enable execution from utils folder

from core.dataset import COCODataset
from core.model import Model

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))

    return grid

def visualize_attention(model, annotations, imagedir, image_indices, outdir=None):
    
    model.extended_output = True

    if outdir is not None:
        pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)

    normal_images = COCODataset(annotations, imagedir, image_size=(224,224))
    input_images = COCODataset(annotations, imagedir, image_size=(224,224), normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225])

    ret = []

    for image_index in tqdm(image_indices):

        context_images, target_images, bbox, _ = input_images[image_index]
        _ , _, _, attention_map = model(context_images.unsqueeze(0), target_images.unsqueeze(0), bbox.unsqueeze(0))
        
        attention_map = attention_map.detach().squeeze() # output dimension: (layers, context_tokens)

        grid_size = int(np.sqrt(attention_map.size(-1)))
        attention_map = attention_map.reshape(attention_map.size(0), grid_size, grid_size)

        layer_visualization = []

        for layer in range(attention_map.size(0)):
            mask = attention_map[layer]
            mask = mask / mask.max() # normalize
            mask = to_tensor(to_pil_image(mask).resize((224,224)))

            layer_visualization.append(to_pil_image(mask * normal_images[image_index][0]))


        visualization = image_grid(layer_visualization, rows=1, cols=len(layer_visualization))
        normal_image = to_pil_image(normal_images[image_index][0])

        # save all visualizations
        if outdir is not None:

            for layer, vis in enumerate(layer_visualization):
                vis.save(os.path.join(outdir, "image_{}_attention_layer_{}.png".format(image_index, layer)))

            visualization.save(os.path.join(outdir, "image_{}_attention_all_layers.png".format(image_index)))
            normal_image.save(os.path.join(outdir, "image_{}_normal.png".format(image_index)))

        ret.append([layer_visualization, visualization, normal_image])

    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, help="Path to config file. If other commmand line arguments are passed in addition to a config, they are used to replace parameters specified in the config.")
    parser.add_argument("--imageindices", type=int, nargs='+', help="Image indices for which to visualize attention. Example usage: --imageindices 1 2 3 10 20")
    parser.add_argument("--outdir", type=str, help="Path to output folder (will be created if it does not exist).")
    parser.add_argument("--annotations", type=str, help="Path to COCO-style annotations file.")
    parser.add_argument("--imagedir", type=str, help="Path to images folder w.r.t. which filenames are specified in the annotations.")
    args = parser.parse_args()

    assert(args.checkpoint is not None), "A checkpoint needs to be specified via commandline argument (--checkpoint)"
    assert(args.config is not None), "A config needs to be specified via commandline argument (--config)"

    with open(args.config) as f:
        cfg = ConfigDict(yaml.load(f, Loader=yaml.Loader))

    cfg.checkpoint = args.checkpoint

    if args.annotations is not None:
        cfg.test_annotations = args.annotations
    if args.imagedir is not None:
        cfg.test_imagedir = args.imagedir

    assert(cfg.test_annotations is not None), "Annotations need to be specified either via commandline argument (--annotations) or config (test_annotations)."
    assert(cfg.test_imagedir is not None), "Imagedir needs to be specified either via commandline argument (--imagedir) or config (test_imagedir)."

    if not hasattr(cfg, "num_classes"): # infer number of classes
        with open(cfg.test_annotations) as f:
            NUM_CLASSES = len(json.load(f)["categories"])
        cfg.num_classes = NUM_CLASSES

    print("Initializing model from checkpoint {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = Model.from_config(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    visualize_attention(model, cfg.test_annotations, cfg.test_imagedir, args.imageindices, args.outdir)