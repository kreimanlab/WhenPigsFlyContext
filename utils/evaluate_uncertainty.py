import os
import sys
import argparse
import datetime
import pathlib
import json
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from ml_collections import ConfigDict

sys.path.append(".") # to enable execution from parent directory
sys.path.append("..") # to enable execution from utils folder

from core.dataset import COCODatasetWithID
from core.config import save_config
from core.model import Model
from core.metrics import DualPredictionLoggerWithID


def evaluate_uncertainty(model, annotations_file, imagedir, savedir=None, outname="uncertainty_evaluation", save_plot=False, epoch=None):
    """
    Logs predictions from both branches, the uncertainty value and the associated ground truth. This can be useful to find a good uncertainty threshold.

    Args:
        annotations_file (str): Path to annotations file.
        imagedir (str): Path to image directory.
        savedir (str, optional): Path to folder where results should be saved. If None, the results are not saved. Defaults to None.
        outname (str, optional): Used to name the output file (and plot if save_plot=True). Defaults to "test".
        save_plot (bool, optional): If set to true, a plot of accuracies for different uncertainty thresholds is saved.
        epoch (int, optional): Can be used to add the epoch to the output file name. Defaults to None.

    Returns an instance of DualPredictionLogger, which holds the required log information.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_state = model.extended_output # to set model back to this state after evaluation
    model.extended_output = True
    model.to(device)

    testset = COCODatasetWithID(annotations_file, imagedir, image_size=(224,224), normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225])
    dataloader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    log = DualPredictionLoggerWithID()
    
    model.eval() # set eval mode
    with torch.no_grad():
        for context_images, target_images, bbox, labels_cpu, annotation_ids in tqdm(dataloader, desc="Test Batches", leave=True):
            context_images = context_images.to(device)
            target_images = target_images.to(device)
            bbox = bbox.to(device)

            output_uncertainty_branch , output_main_branch, uncertainty, _ = model(context_images, target_images, bbox)
            
            _, predictions_uncertainty_branch = torch.max(output_uncertainty_branch.detach().to("cpu"), 1) # choose idx with maximum score as prediction
            _, predictions_main_branch = torch.max(output_main_branch.detach().to("cpu"), 1) # choose idx with maximum score as prediction
            log.update(predictions_uncertainty_branch, predictions_main_branch, uncertainty, labels_cpu, annotation_ids)

    model.extended_output = model_state # set back to original state

    # save
    if savedir is not None:
        pathlib.Path(savedir).mkdir(exist_ok=True, parents=True)

        if epoch is not None:
            log.save(savedir, name="{}_epoch_{}".format(outname, epoch))
        else:
            log.save(savedir, name=outname)

        if save_plot:
            fig = log.plot_accuracy_vs_threshold()
            fig.savefig(os.path.join(savedir, "{}_plot.png".format(outname)))

    return log


if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, help="Path to config file. If other commmand line arguments are passed in addition to a config, they are used to replace parameters specified in the config.")
    parser.add_argument("--outdir", type=str, default="evaluation/uncertainty_evaluation_{date:%Y-%m-%d_%H%M}".format(date=datetime.datetime.now()), help="Path to output folder (will be created if it does not exist).")
    parser.add_argument("--outname", type=str, default="uncertainty_evaluation", help="Name used for the output file: outname.json, outname_plot.json")

    parser.add_argument("--annotations", type=str, help="Path to COCO-style annotations file.")
    parser.add_argument("--imagedir", type=str, help="Path to images folder w.r.t. which filenames are specified in the annotations.")
    
    parser.add_argument("--uncertainty_gate_type", type=str, help="Uncertainty gate type to use.")
    parser.add_argument("--weighted_prediction", action='store_true', dest='weighted_prediction', help="If set, the model outputs a weighted prediction if the uncertainty gate prediction exceeds the uncertainty threshold.")
    parser.add_argument("--unweighted_prediction", action='store_false', dest='weighted_prediction', help="If set, the model outputs unweighted predictions.")    
    parser.set_defaults(weighted_prediction=None)

    parser.add_argument("--save_plot", action='store_true', default=False, help="Set to save plot of accuracies at different uncertainty thresholds.")
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
    if args.uncertainty_gate_type is not None:
        cfg.uncertainty_gate_type = args.uncertainty_gate_type
    if args.weighted_prediction is not None:
        cfg.weighted_prediction = args.weighted_prediction

    assert(cfg.test_annotations is not None), "Annotations need to be specified either via commandline argument (--annotations) or config (test_annotations)."
    assert(cfg.test_imagedir is not None), "Imagedir needs to be specified either via commandline argument (--imagedir) or config (test_imagedir)."

    if not hasattr(cfg, "num_classes"): # infer number of classes
        with open(cfg.annotations) as f:
            NUM_CLASSES = len(json.load(f)["categories"])
        cfg.num_classes = NUM_CLASSES

    pathlib.Path(args.outdir).mkdir(exist_ok=True, parents=True)
    save_config(cfg, args.outdir)
    print(cfg)

    print("Initializing model from checkpoint {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = Model.from_config(cfg, extended_output=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    assert not missing_keys, "Checkpoint is missing keys required to initialize the model: {}".format(missing_keys)
    if len(unexpected_keys):
        print("Checkpoint contains unexpected keys that were not used to initialize the model: ")
        print(unexpected_keys)

    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    evaluate_uncertainty(model, cfg.test_annotations, cfg.test_imagedir, args.outdir, outname=args.outname, save_plot=args.save_plot)
