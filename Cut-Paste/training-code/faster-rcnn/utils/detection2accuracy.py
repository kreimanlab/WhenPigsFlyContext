"""
Evaluation of classification accuracy. We take the predictions of an object detection method and
extract the bounding box with the highest overlap and confidence. Then we compare the class label
for this box with the true label.
"""

import os
import pathlib
import json
import numpy as np
import pandas as pd

from collections import defaultdict

from .utils import computeIOU


def detection2accuracy(detections, groundtruth, outdir, record_individual_scores=False):
    """
    Writes class accuracies and total accuracy (first averaged within classes, then across classes) to a file in outdir.

    Arguments:
        detections: path to coco_instances_results.json file
        groundtruth: path to the ground truth annotations file (json file with COCO-style annotations)
        outdir: directory in which the results should be saved
    """

    with open(detections) as f:
        results = pd.DataFrame(json.load(f))

    with open(groundtruth) as f:
        coco_dict = json.load(f)
        gt = coco_dict["annotations"]

    # id2label mapping
    id2label = {c["id"]: c["name"] for c in coco_dict["categories"]}

    # select best overlapping bounding-box for each ground truth annotation. For ties use confidences and if still tied, average.
    iouthreshold = 0.9 # what counts as good bounding box
    nmatches = 0
    top1 = defaultdict(list)

    if record_individual_scores:
        individual_scores = []

    for gt_annotation in gt:
        candidates = results[results.image_id == gt_annotation["image_id"]]
        if len(candidates) == 0: # no bboxes predicted for this image
            top1[gt_annotation["category_id"]].append(0)
            if record_individual_scores:
                individual_scores.append([gt_annotation["id"], int(results.iloc[max_ind]["category_id"] == gt_annotation["category_id"]), gt_annotation["category_id"], id2label[gt_annotation["category_id"]], None, None])
            continue

        ious = candidates.bbox.apply(lambda x: computeIOU(gt_annotation["bbox"], x))
        if ious.max() > iouthreshold: # count how many good bboxes are predicted
            nmatches += 1
        
        max_ind = ious.idxmax()
        if type(max_ind) == pd.Series: # more than one bbox has maximum IOU
            max_ind = candidates.iloc[max_ind].score.idxmax() # select box with higher confidence
        if type(max_ind) == pd.Series: #more than one bbox with same maximum IOU and same score: average the predictions
            top1[gt_annotation["category_id"]].append(np.mean(results.iloc[max_ind]["category_id"] == gt_annotation["category_id"]))
            if record_individual_scores:
                individual_scores.append([gt_annotation["id"], int(results.iloc[max_ind]["category_id"] == gt_annotation["category_id"]), gt_annotation["category_id"], id2label[gt_annotation["category_id"]], int(results.iloc[max_ind]["category_id"]), id2label[results.iloc[max_ind]["category_id"]]])
        else:
            top1[gt_annotation["category_id"]].append(results.iloc[max_ind]["category_id"] == gt_annotation["category_id"]) 
            if record_individual_scores:
                individual_scores.append([gt_annotation["id"], int(results.iloc[max_ind]["category_id"] == gt_annotation["category_id"]), gt_annotation["category_id"], id2label[gt_annotation["category_id"]], int(results.iloc[max_ind]["category_id"]), id2label[results.iloc[max_ind]["category_id"]]])


    print("Corresponding bounding boxes with IOU > {} were found for {} out of {} target objects.\n".format(iouthreshold, nmatches, len(gt)))

    named_class_accuracies = {}
    for class_id, val in top1.items():
        acc = np.mean(val)
        print("Class: {:>5} \t Accuracy: {}".format(id2label[class_id], acc))
        named_class_accuracies[id2label[class_id]] = acc
    
    total_accuracy = float(np.mean(list(named_class_accuracies.values())))
    print("\nTotal Accuracy: {}".format(total_accuracy))
    
    savedict = {"total_accuracy": total_accuracy,
                "named_class_accuracies": named_class_accuracies}

    pathlib.Path(outdir).mkdir(exist_ok=True) # create outdir if it does not exist yet
    with open(os.path.join(outdir, "accuracies.json"), "w") as f:
        json.dump(savedict, f)

    if record_individual_scores:
        with open(os.path.join(outdir, "individual_scores.json"), "w") as f:
            json.dump(individual_scores, f)


if __name__ == "__main__":
    detections = "../evaluation/train_with_gt_bboxes/test_on_unrel/coco_instances_results.json"
    groundtruth = "/media/data/philipp_data/UnRel_test/annotations/annotations.json"
    outdir = "../evaluation/train_with_gt_bboxes/test_on_unrel/"

    detection2accuracy(detections, groundtruth, outdir)
