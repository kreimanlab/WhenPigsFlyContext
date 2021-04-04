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

from core.dataset import COCODatasetWithID
from core.config import save_config
from core.model import Model
from core.metrics import AccuracyLogger, IndividualScoreLogger


def test(model, annotations_file, imagedir, outdir, outname="test", epoch=None, record_individual_scores=False, print_batch_metrics=False):
    """
    Arguments:
        epoch: If specified, it is used to include the epoch in the output file name.
    """
    pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    testset = COCODatasetWithID(annotations_file, imagedir, image_size=(224,224), normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225])
    dataloader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    if print_batch_metrics:
        criterion = nn.CrossEntropyLoss()

    test_accuracy = AccuracyLogger(testset.idx2label)

    if record_individual_scores:
        individual_scores = IndividualScoreLogger(testset.idx2label)
    
    model.eval() # set eval mode
    with torch.no_grad():
        for i, (context_images, target_images, bbox, labels_cpu, annotation_ids) in enumerate(tqdm(dataloader, desc="Test Batches", leave=True)):
            context_images = context_images.to(device)
            target_images = target_images.to(device)
            bbox = bbox.to(device)
            labels = labels_cpu.to(device) # keep a copy of labels on cpu to avoid unnecessary transfer back to cpu later

            output = model(context_images, target_images, bbox) # output is (batchsize, num_classes) tensor of logits
            _, predictions = torch.max(output.detach().to("cpu"), 1) # choose idx with maximum score as prediction
            test_accuracy.update(predictions, labels_cpu)

            if record_individual_scores:
                individual_scores.update(predictions.to("cpu"), labels_cpu, annotation_ids)

            # print
            if print_batch_metrics:
                batch_loss = criterion(output, labels).item()
                batch_corr = sum(predictions == labels_cpu) # number of correct predictions
                batch_accuracy = batch_corr # / batch_size # since batchsize is 1

                print("\t Test Batch {}: \t Loss: {} \t Accuracy: {}".format(i, batch_loss, batch_accuracy))
        
    print("\nTotal Test Accuracy: {}".format(test_accuracy.accuracy()))
    print("{0:20} {1:10}".format("Class", "Accuracy")) # header
    for name, acc in test_accuracy.named_class_accuarcies().items():
        print("{0:20} {1:10.4f}".format(name, acc))

    # save accuracies
    if epoch is not None:
        test_accuracy.save(outdir, name="{}_accuracies_epoch_{}".format(outname, epoch))
    else:
        test_accuracy.save(outdir, name="{}_accuracies".format(outname))

    if record_individual_scores:
        individual_scores.save(outdir, name="{}_individual_scores".format(outname))

    return test_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--checkpoint", type=str, default="/home/dimitar/BigPictureNet/output/2021-03-10_2335/checkpoint_08.tar", help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, default="/home/dimitar/BigPictureNet/output/2021-03-10_2335/config.yaml", help="Path to config file. If other commmand line arguments are passed in addition to a config, they are used to replace parameters specified in the config.")
    parser.add_argument("--outdir", type=str, default="evaluation/{date:%Y-%m-%d_%H%M}".format(date=datetime.datetime.now()), help="Path to output folder (will be created if it does not exist).")
    parser.add_argument("--outname", type=str, default="test_exp_I", help="Name used for the output file: outname_accuracies.json, outname_individual_scores.json")

    parser.add_argument("--annotations", type=str, default='/home/dimitar/experiments_I_and_J/annotations/test_annotations_exp_I.json',  help="Path to COCO-style annotations file.")
    parser.add_argument("--imagedir", type=str, default="/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expA", help="Path to images folder w.r.t. which filenames are specified in the annotations.")
    parser.add_argument("--uncertainty_threshold", type=float, help="Uncertainty threshold for the uncertainty gating module.")
    parser.add_argument("--weighted_prediction", action='store_true', dest='weighted_prediction', help="If set, the model outputs a weighted prediction if the uncertainty gate prediction exceeds the uncertainty threshold.")
    parser.add_argument("--unweighted_prediction", action='store_false', dest='weighted_prediction', help="If set, the model outputs unweighted predictions.")    
    parser.set_defaults(weighted_prediction=None)

    parser.add_argument("--record_individual_scores", action='store_true', default=True, help="If set, will log for each individual annotion how it was predicted and if the prediction was correct")
    parser.add_argument("--print_batch_metrics", action='store_true', default=False, help="Set to print metrics for every batch.")
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
    if args.uncertainty_threshold is not None:
        cfg.uncertainty_threshold = args.uncertainty_threshold
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
    model = Model.from_config(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    test(model, cfg.test_annotations, cfg.test_imagedir, args.outdir, outname=args.outname, record_individual_scores=args.record_individual_scores , print_batch_metrics=args.print_batch_metrics)
