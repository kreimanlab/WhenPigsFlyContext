import os
import warnings
import argparse
import datetime
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from test import test
from utils.evaluate_uncertainty import evaluate_uncertainty
from core.config import create_config, save_config
from core.dataset import COCODataset
from core.model import Model
from core.metrics import AccuracyLogger


## Initialization
#
    
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file. If additional commandline options are provided, they are used to modify the specifications in the config file.")
parser.add_argument("--outdir", type=str, default="output/{date:%Y-%m-%d_%H%M}".format(date=datetime.datetime.now()), help="Path to output folder (will be created if it does not exist).")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint from which to continue training.")
parser.add_argument("--annotations", type=str, help="Path to COCO-style annotations file.")
parser.add_argument("--imagedir", type=str, help="Path to images folder w.r.t. which filenames are specified in the annotations.")

parser.add_argument("--test_annotations", type=str, help="Path to COCO-style annotations file for model evaluation.")
parser.add_argument("--test_imagedir", type=str, help="Path to images folder w.r.t. which filenames are specified in the annotations for model evaluation.")
parser.add_argument("--test_frequency", type=int, default=1, help="Evaluate model on test data every __ epochs.")

parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
parser.add_argument("--save_frequency", type=int, default=1, help="Save model checkpoint every __ epochs.")
parser.add_argument("--print_batch_metrics", action='store_true', default=False, help="Set to print metrics for every batch.")

parser.add_argument("--batch_size", type=int, help="Batchsize to use for training.")
parser.add_argument("--learning_rate", type=float, help="Learning rate to use for training.")
parser.add_argument("--imbalance_reweighting", action='store_true', help="Reweight samples in proportion to the number of samples per class.")
parser.add_argument("--num_decoder_heads", type=int, help="Number of decoder heads.")
parser.add_argument("--num_decoder_layers", type=int, help="Number of decoder layers.")
parser.add_argument("--uncertainty_gate_type", type=str, help="Uncertainty gating mechanism to use. Can be one of: 'entropy', 'relative_softmax_distance', 'learned', 'learned_metric'.")
parser.add_argument("--uncertainty_threshold", type=float, help="Uncertainty threshold for the uncertainty gating module. Note that training does not depend on the threshold, the model can still be used with different thresholds later.")
parser.add_argument("--weighted_prediction", action='store_true', default=None, help="If enabled, the model returns an uncertainty-weighted prediction if the uncertainty_gate prediction exceeds the uncertainty threshold.")
args = parser.parse_args()

# Create output directory
pathlib.Path(args.outdir).mkdir(exist_ok=True, parents=True)

# Load config or create a new one
cfg = create_config(args)

dataset = COCODataset(cfg.annotations, cfg.imagedir, image_size =(224,224), normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225])
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

NUM_CLASSES = dataset.NUM_CLASSES
cfg.num_classes = NUM_CLASSES
save_config(cfg, args.outdir)
print(cfg)

model = Model.from_config(cfg)

assert(model.TARGET_IMAGE_SIZE == model.CONTEXT_IMAGE_SIZE == dataset.image_size), "Image size from the dataset is not compatible with the encoder."

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

if cfg.imbalance_reweighting:
    class_weights = torch.true_divide(dataset.relative_annotation_counts.max(), dataset.relative_annotation_counts)
    criterion = nn.CrossEntropyLoss(weight= class_weights.to(device))
else:
    criterion = nn.CrossEntropyLoss()

if cfg.checkpoint is not None:
    print("Initializing from checkpoint {}".format(cfg.checkpoint))
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print("No checkpoint was passed.")
    model.to(device)
    start_epoch = 1

# Tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.outdir, "runs/{date:%Y-%m-%d_%H%M}".format(date=datetime.datetime.now())))
context_images, target_images, bbox, labels = iter(dataloader).next()
writer.add_images("context_image_batch", context_images) # add example context image batch to tensorboard log
writer.add_images("target_image_batch", target_images) # add example target image batch to tensorboard log
with warnings.catch_warnings(): # add_graph method is known to issue a warning
    warnings.simplefilter("ignore")
    writer.add_graph(model, input_to_model=[context_images.to(device), target_images.to(device), bbox.to(device)]) # add model graph to tensorboard log

accuracy_logger_main_branch = AccuracyLogger(dataset.idx2label)
accuracy_logger_uncertainty_branch = AccuracyLogger(dataset.idx2label)


## Training
#

for epoch in tqdm(range(start_epoch, args.epochs + 1), position=0, desc="Epochs", leave=True):

    model.train() # set train mode
    accuracy_logger_main_branch.reset() # reset accuracy logger every epoch
    accuracy_logger_uncertainty_branch.reset()

    for i, (context_images, target_images, bbox, labels_cpu) in enumerate(tqdm(dataloader, position=1, desc="Batches", leave=True)):
        context_images = context_images.to(device)
        target_images = target_images.to(device)
        bbox = bbox.to(device)
        labels = labels_cpu.to(device) # keep a copy of labels on cpu to avoid unnecessary transfer back to cpu later

        output_uncertainty_branch , output_main_branch, output_weighted, uncertainty = model(context_images, target_images, bbox)

        # backpropagation through both branches
        optimizer.zero_grad(set_to_none=True)

        if cfg.uncertainty_gate_type == "learned" or cfg.uncertainty_gate_type == "learned_metric":
            loss_uncertainty_estimator = criterion(output_weighted, labels)
            loss_uncertainty_estimator.backward(retain_graph=True)    

        loss_uncertainty_branch = criterion(output_uncertainty_branch, labels)
        loss_uncertainty_branch.backward(retain_graph=True)

        loss_main_branch = criterion(output_main_branch, labels)
        loss_main_branch.backward()

        optimizer.step()
        
        # log metrics
        _, predictions_uncertainty_branch = torch.max(output_uncertainty_branch.detach().to("cpu"), 1) # choose idx with maximum score as prediction
        batch_accuracy_uncertainty_branch = sum(predictions_uncertainty_branch == labels_cpu) / cfg.batch_size
        batch_loss_uncertainty_branch = loss_uncertainty_branch.item()
        writer.add_scalar("Batch Accuracy Uncertainty Branch/train", batch_accuracy_uncertainty_branch, i + (epoch - 1) * len(dataloader))
        writer.add_scalar("Batch Loss Uncertainty Branch/train", batch_loss_uncertainty_branch, i + (epoch - 1) * len(dataloader))
        accuracy_logger_uncertainty_branch.update(predictions_uncertainty_branch, labels_cpu)

        _, predictions_main_branch = torch.max(output_main_branch.detach().to("cpu"), 1) # choose idx with maximum score as prediction
        batch_accuracy_main_branch = sum(predictions_main_branch == labels_cpu) / cfg.batch_size
        batch_loss_main_branch = loss_main_branch.item()
        writer.add_scalar("Batch Accuracy Main Branch/train", batch_accuracy_main_branch, i + (epoch - 1) * len(dataloader))
        writer.add_scalar("Batch Loss Main Branch/train", batch_loss_main_branch, i + (epoch - 1) * len(dataloader))
        accuracy_logger_main_branch.update(predictions_main_branch, labels_cpu)

        writer.add_scalar("Batch Uncertainty/train", torch.mean(uncertainty), i + (epoch - 1) * len(dataloader))

        if args.print_batch_metrics:
            print("\t Epoch {}, Batch {}: \t Loss: {} \t Accuracy: {}".format(epoch, i, batch_loss_main_branch, batch_accuracy_main_branch))


    # log metrics
    writer.add_scalar("Total Accuracy Main Branch/train", accuracy_logger_main_branch.accuracy(), epoch * len(dataloader))
    writer.add_scalar("Total Accuracy Uncertainty Branch/train", accuracy_logger_uncertainty_branch.accuracy(), epoch * len(dataloader))

    print("\nEpoch {}, Train Accuracy: {}".format(epoch, accuracy_logger_main_branch.accuracy()))
    print("{0:20} {1:10}".format("Class", "Accuracy")) # header
    for name, acc in accuracy_logger_main_branch.named_class_accuarcies().items():
        writer.add_scalar("Class Accuracies Main Branch/train/{}".format(name), acc, epoch * len(dataloader))
        print("{0:20} {1:10.4f}".format(name, acc))

    for name, acc in accuracy_logger_uncertainty_branch.named_class_accuarcies().items():
        writer.add_scalar("Class Accuracies Uncertainty Branch/train/{}".format(name), acc, epoch * len(dataloader))

    # save checkpoint and training accuracies
    if epoch % args.save_frequency == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.outdir + "/checkpoint_{}.tar".format(epoch))
        print("Checkpoint saved.")

        accuracy_logger_main_branch.save(args.outdir, name="train_accuracies_epoch_{}".format(epoch))
        accuracy_logger_uncertainty_branch.save(args.outdir, name="train_accuracies_uncertainty_branch_epoch_{}".format(epoch))
    
    # evaluation on test data
    if cfg.test_annotations is not None and cfg.test_imagedir is not None and epoch % args.test_frequency == 0:
        print("Starting evaluation on test data.")
        test_accuracy = test(model, cfg.test_annotations, cfg.test_imagedir, outdir=args.outdir, epoch=epoch)

        writer.add_scalar("Total Accuracy/test", test_accuracy.accuracy(), epoch * len(dataloader))
        for name, acc in test_accuracy.named_class_accuarcies().items():
            writer.add_scalar("Class Accuracies/test/{}".format(name), acc, epoch * len(dataloader))

        print("Starting uncertainty evaluation.")
        test_uncertainty_log = evaluate_uncertainty(model, cfg.test_annotations, cfg.test_imagedir)
        writer.add_figure("Uncertainty Threshold Curve", test_uncertainty_log.plot_accuracy_vs_threshold(), epoch * len(dataloader))

        if (args.epochs - epoch) / args.test_frequency < 1: # last evaluation
            writer.add_hparams({"learning_rate": cfg.learning_rate, "num_decoder_layers": cfg.num_decoder_layers, "num_decoder_heads": cfg.num_decoder_heads,
                                "uncertainty_gate_type": cfg.uncertainty_gate_type, "uncertainty_threshold": cfg.uncertainty_threshold, "imbalance_reweighting": str(cfg.imbalance_reweighting)},
                                metric_dict={"hparam/accuracy": test_accuracy.accuracy()})
        
writer.close()
