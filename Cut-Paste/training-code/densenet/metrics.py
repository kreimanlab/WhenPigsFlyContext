import os
import json

import torch

from collections import defaultdict

class AccuracyLogger():
    """
    Logs per-class accuracies and total accuracy.
    """

    def __init__(self, idx2label_dict, device=None):
        """
        Parameters:
            idx2label_dict: dictionary mapping class index to class name
            writer: a torch.utils.tensorboard.SummaryWriter instance that is used create tensorboard logs. If None is passed, tensorboard logging is skipped.
            device: device on which the logger will be used. E.g., "cuda:0" if you want to use it with tensors on the GPU.
        """
        self.idx2label = idx2label_dict
        self.NUM_CLASSES = len(idx2label_dict)

        self.confusionmatrix = torch.zeros(self.NUM_CLASSES, self.NUM_CLASSES, device=device) # confusionmatrix[i,j] holds the number of samples with true label i that were classified as j.
        self.distributions = torch.zeros(self.NUM_CLASSES, self.NUM_CLASSES, device=device) # similar to confusionmatrix but we sum the softmax distributions instead of "point" predictions.

    @classmethod
    def from_state_dict(cls, state_dict, device=None):
        """
        Initializes an AccuracyLogger from a state_dict.
        """
        ret = cls(state_dict["idx2label_dict"], device)
        ret.confusionmatrix = state_dict["confusionmatrix"]
        ret.distributions = state_dict["distributions"]
        return ret

    def reset(self):
        """
        Resets the logged statistics to 0.
        """
        self.confusionmatrix *= 0
        self.distributions *= 0

    def update(self, output, groundtruth):
        """
        Parameters:
            output: model output with (batchsize, num_classes) tensor of logits
            groundtruth: groundtruth class indexes corresponding to the predictions.
        """
        _, predictions = torch.max(output, 1) # choose idx with maximum score as prediction
        output_normalized = normalize_scores(output)

        assert(len(predictions) == len(groundtruth)), "Predictions and groundtruth should be of the same length."

        for i, (pred, gt) in enumerate(zip(predictions, groundtruth)):
            self.confusionmatrix[gt, pred] += 1
            self.distributions[gt] += output_normalized[i]

    def accuracy(self):
        """
        Returns the total accuracy (first averaged within class and then across classes).
        """
        return (self.confusionmatrix.diag()/(self.confusionmatrix.sum(dim=1) + 1e-7)).mean().item()

    def class_accuracies(self):
        """
        returns a dictionary mapping class ind to accuracy.
        """
        return {idx: acc.item() for idx, acc in enumerate(self.confusionmatrix.diag()/(self.confusionmatrix.sum(dim=1) + 1e-7))}

    def named_class_accuarcies(self):
        """
        returns a dictionary mapping class name to accuracy.
        """
        return {self.idx2label[idx]: acc.item() for idx, acc in enumerate(self.confusionmatrix.diag()/(self.confusionmatrix.sum(dim=1) + 1e-7))}

    def normalized_confusionmatrix(self):
        """
        returns the normalized confusion matrix (as tensor): confusionmatrix[i,j] holds the percentage of classes with real label i that were classified as class j 
        """

        return (self.confusionmatrix.transpose(0,1) / (self.confusionmatrix.sum(dim=1) + 1e-7)).transpose(0,1)

    def normalized_distributions(self):
        """
        returns the tensor of normalized prediction distributions: row i is the average softmax prediction vector (averaged over all samples with real label i).
        """
        return (self.distributions.transpose(0,1) / (self.confusionmatrix.sum(dim=1) + 1e-7)).transpose(0,1)

    def state_dict(self):
        """
        Returns a state dict that can be used to restore the AccuracyLogger's state with the from_state_dict class method.
        """
        return {"idx2label_dict": self.idx2label, "confusionmatrix": self.confusionmatrix, "distributions": self.distributions}

    def save(self, savedir, name="accuracies"):
        """
        Saves logged information to "savedir/name.json".
        """
        savedict = {"total_accuracy": self.accuracy(),
                    "class_accuracies": self.class_accuracies(),
                    "named_class_accuracies": self.named_class_accuarcies(),
                    "confusionmatrix": self.confusionmatrix.tolist(),
                    "normalized_confusionmatrix": self.normalized_confusionmatrix().tolist(),
                    "distributions": self.distributions.tolist(),
                    "normalized_distributions": self.normalized_distributions().tolist()}
        with open(os.path.join(savedir, name +".json"), "w") as f:
            json.dump(savedict, f)



class IndividualScoreLogger():
    """
    Keeps track of individual scores: stores for every annotation what the predicted label was and if it was correct.
    The log field holds a list, where each entry is itself a list consisting of [annotation_id, correct (bool), prediction, prediction label name]
    """

    def __init__(self, idx2label_dict):
        self.idx2label = idx2label_dict
        self.NUM_CLASSES = len(idx2label_dict)
        self.log = []

    def update(self, output, groundtruth, annotation_ids):
        _, predictions = torch.max(output, 1) # choose idx with maximum score as prediction

        assert(len(predictions) == len(groundtruth) and len(predictions) == len(annotation_ids)), "Predictions, groundtruth and annotation_ids should be of the same length."

        for pred, gt, annotation_id in zip(predictions, groundtruth, annotation_ids):
            self.log.append([annotation_id.item(), (pred == gt).item(), gt.item(), self.idx2label[gt.item()], pred.item(), self.idx2label[pred.item()]]) # [annotation_id, correct, gt_id, gt_label, pred_id, pred_label]
        
    def save(self, savedir, name="individual_scores"):
        """
        Saves logged information to "savedir/name.json".
        """
        with open(os.path.join(savedir, name +".json"), "w") as f:
            json.dump(self.log, f)


def normalize_scores(scores):
    """
    Returns scores scaled to [0, 1] while retaining the relative differences among the entires (unlike softmax).

    Parameters:
        scores: tensor of size (batchsize, num_classes)
    """
    ret = scores - scores.min(1, keepdim=True)[0]
    ret /= ret.max(1, keepdim=True)[0]
    return  ret