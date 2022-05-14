import argparse
import pathlib

import torch, torchvision
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils.dataset import COCODataset, COCODatasetWithID
from utils.metrics import AccuracyLogger, IndividualScoreLogger

def test(model, annotations_file, image_dir, image_size, output_dir, epoch=None, record_individual_scores=False, print_batch_metrics=False):
    """
    Arguments:
        epoch: If specified, it is used to include the epoch in the output file name.
    """
    pathlib.Path(output_dir).mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    testset = COCODatasetWithID(annotations_file, image_dir, image_size, normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225])    
    dataloader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    if print_batch_metrics:
        criterion = nn.CrossEntropyLoss()

    test_accuracy = AccuracyLogger(testset.idx2label, device=device)

    if record_individual_scores:
        individual_scores = IndividualScoreLogger(testset.idx2label)
    
    model.eval() # set eval mode
    with torch.no_grad():
        for i, (batch_inputs, batch_labels, annotation_ids) in enumerate(tqdm(dataloader, desc="Test Batches", leave=True)):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_inputs) # output is (batchsize, num_classes) tensor of logits
            test_accuracy.update(output, batch_labels)

            if record_individual_scores:
                individual_scores.update(output.to("cpu"), batch_labels.to("cpu"), annotation_ids)

            # print
            if print_batch_metrics:
                batch_loss = criterion(output, batch_labels).item()
                _, predictions = torch.max(output, 1) # choose idx with maximum score as prediction
                batch_corr = sum(predictions == batch_labels) # number of correct predictions
                batch_accuracy = batch_corr # / batch_size # since batchsize is 1

                print("\t Test Batch {}: \t Loss: {} \t Accuracy: {}".format(i, batch_loss, batch_accuracy))
        
    print("\nTotal Test Accuracy: {}".format(test_accuracy.accuracy()))
    print("{0:20} {1:10}".format("Class", "Accuracy")) # header
    for name, acc in test_accuracy.named_class_accuarcies().items():
        print("{0:20} {1:10.4f}".format(name, acc))

    # save accuracies
    if epoch is not None:
        test_accuracy.save(output_dir, name="test_accuracies_epoch_{}".format(epoch))
    else:
        test_accuracy.save(output_dir, name="test_accuracies_epoch_10_exp_J")

    if record_individual_scores:
        individual_scores.save(output_dir, name="individual_scores_epoch_10_exp_J")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="exp_IJ_output", help="Path to output folder (will be created if it does not exist).")
    parser.add_argument("--checkpoint", type=str, default="/home/dimitar/densenet-master/output/checkpoint_10.tar", help="Path to model checkpoint.")
    parser.add_argument("--annotations_file", type=str, default='/home/dimitar/experiments_I_and_J/annotations/test_annotations_exp_J.json', help="Path to COCO-style annotations file.")
    parser.add_argument("--image_dir", type=str, default="/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expH", help="Path to images folder.")
    parser.add_argument("--image_size", type=tuple, default=(224, 224), help="Input image size the model requires.")
    parser.add_argument("--num_classes", type=int, default=55, help="Number of classes.")
    parser.add_argument('--record_individual_scores', action='store_true', default=True, help="If set, will log for each individual annotion how it was predicted and if the prediction was correct")
    parser.add_argument("--print_batch_metrics", action='store_true', default=False, help="Set to print metrics for every batch.")
    args = parser.parse_args()

    assert(args.checkpoint is not None), "No checkpoint was passed. A checkpoint is required to load the model."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Initializing model from checkpoint {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)

    model = torchvision.models.densenet169()
    model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test(model, args.annotations_file, args.image_dir, args.image_size, args.output_dir, record_individual_scores=args.record_individual_scores , print_batch_metrics=args.print_batch_metrics)
