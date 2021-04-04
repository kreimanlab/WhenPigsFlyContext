import argparse
import pathlib

import torch, torchvision
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from test import test
from utils.dataset import COCODataset
from utils.metrics import AccuracyLogger


## Initialization
#

NUM_CLASSES = 55
image_size = (224, 224) # for DenseNet


# Checkpoints: "/home/dimitar/densenet-master/output/checkpoint_x.tar"

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output", help="Path to output folder (will be created if it does not exist).")
parser.add_argument("--checkpoint", type=str, default="/home/dimitar/densenet-master/output/checkpoint_10.tar", help="Path to model checkpoint.")
parser.add_argument("--annotations_file", type=str, default="/home/dimitar/train_annotations.json", help="Path to COCO-style annotations file.")
parser.add_argument("--image_dir", type=str, default="/home/mengmi/Projects/Proj_context2/Datasets/MSCOCO/trainColor_oriimg", help="Path to images folder.")
parser.add_argument("--test_annotations_file", type=str, default='/home/dimitar/test_annotations.json', help="Path to COCO-style annotations file for model evaluation.")
parser.add_argument("--test_image_dir", type=str, default='/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expH', help="Path to images folder for model evaluation.")
parser.add_argument("--test_frequency", type=int, default=1, help="Evaluate model on test data every __ epochs.")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
parser.add_argument("--start_epoch", type=int, default=1, help="From which epoch to start training. If a checkpoint is used, it is inferred from the checkpoint.")
parser.add_argument("--batch_size", type=int, default=32, help="Batchsize to use for training.")
parser.add_argument("--print_batch_metrics", action='store_true', default=False, help="Set to print metrics for every batch.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize logger
writer = SummaryWriter(log_dir=args.output_dir)

# create output directory
pathlib.Path(args.output_dir).mkdir(exist_ok=True)

# initialize dataset and dataloader
dataset = COCODataset(args.annotations_file, args.image_dir, image_size, normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225])
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

assert(NUM_CLASSES == dataset.NUM_CLASSES), "Number of classes differs in the model specification ({}) and dataset ({}).".format(NUM_CLASSES, dataset.NUM_CLASSES)

# initialize model, optimizer, metrics logger
if args.checkpoint is not None:
    print("Initializing model from checkpoint {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)

    model = torchvision.models.densenet169()
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    accuracy_logger = AccuracyLogger.from_state_dict(checkpoint['accuracy_logger_state_dict'], device=device)
    accuracy_logger.distributions.requires_grad = False

    args.start_epoch = checkpoint['epoch'] + 1
else:
    print("Initializing model from torchvision.")
    model = torchvision.models.densenet169(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.to(device)

    # TODO: could use an initializer for untrained weights

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # initialize metric logger
    accuracy_logger = AccuracyLogger(dataset.idx2label, device=device)

# set criterion (loss function)
criterion = nn.CrossEntropyLoss()


## Training
#

for epoch in tqdm(range(args.start_epoch, args.epochs + 1), position=0, desc="Epochs", leave=True):

    model.train() # set train mode
    accuracy_logger.reset() # reset accuracy logger

    for i, (batch_inputs, batch_labels) in enumerate(tqdm(dataloader, position=1, desc="Batches", leave=True)):
#    Debugging
#    dl = iter(dataloader)
#    for i in range(20):
#        batch_inputs, batch_labels = next(dl)
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        output = model(batch_inputs) # output is (batchsize, num_classes) tensor of logits
        _, predictions = torch.max(output, 1) # choose idx with maximum score as prediction
        loss = criterion(output, batch_labels)
        loss.backward()

        optimizer.step()

        # log metrics
        batch_loss = loss.item()
        batch_corr = sum(predictions == batch_labels) # number of correct predictions
        batch_accuracy = batch_corr / args.batch_size

        writer.add_scalar("Loss/train", batch_loss, i + (epoch - 1) * len(dataloader))
        writer.add_scalar("Accuracy/train", batch_accuracy, i + (epoch - 1) * len(dataloader))

        accuracy_logger.update(output.detach(), batch_labels)

        # print
        if args.print_batch_metrics:
            print("\t Epoch {}, Batch {}: \t Loss: {} \t Accuracy: {}".format(epoch, i, batch_loss, batch_accuracy))


    # print metrics
    print("\nEpoch {}, Train Accuracy: {}".format(epoch, accuracy_logger.accuracy()))
    print("{0:20} {1:10}".format("Class", "Accuracy")) # header
    for name, acc in accuracy_logger.named_class_accuarcies().items():
        print("{0:20} {1:10.4f}".format(name, acc))

    # save checkpoint
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'accuracy_logger_state_dict': accuracy_logger.state_dict()}, args.output_dir + "/checkpoint_{}.tar".format(epoch))
    print("Checkpoint saved.")

    # save accuracies
    accuracy_logger.save(args.output_dir, name="train_accuracies_epoch_{}".format(epoch))
    
    # evaluation on test data
    if args.test_annotations_file is not None and args.test_image_dir is not None and epoch % args.test_frequency == 0:
        print("Starting evaluation on test data.")
        test(model, args.test_annotations_file, args.test_image_dir, image_size, args.output_dir, epoch=epoch)
