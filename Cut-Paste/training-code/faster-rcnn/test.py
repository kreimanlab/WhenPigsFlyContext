import os
import pathlib
import glob
import json
import torch, torchvision
import detectron2

import numpy as np

from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format


## Configuration
#

outdir = "evaluation/train_with_gt_bboxes/test_on_unrel" # set None to use cfg.OUTPUT_DIR
last_only = True # if True, only the final model is evaluated, if False, all model checkpoints in output are evaluated
use_gt_bboxes = True  # If True, the region proposal step of the model is skipped and the ROI head receives the bounding boxes from the ground truth.
                      # We use this to help the model when we only want to evaluate the classification without the localization step.

# set test dataset
dataset = "UnRel" # "COCOstuff_val"
#register_coco_instances(dataset, {}, "/media/data/philipp_data/COCOstuff/annotations/val.json", "/media/data/philipp_data/COCOstuff/images/val")
#register_coco_instances(dataset, {}, "/media/data/philipp_data/COCOstuff/annotations_UnRel_compatible/val.json", "/media/data/philipp_data/COCOstuff/images/val")
register_coco_instances(dataset, {}, "/media/data/philipp_data/UnRel_test/annotations/annotations.json", "/media/data/philipp_data/UnRel_test/images")

# load model
cfg = get_cfg() # to load a default cfg: cfg = model_zoo.get_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", trained=True)
with open("output/model.yaml") as f:
    cfg = cfg.load_cfg(f)

print("Numer of classes: {}".format(cfg.MODEL.ROI_HEADS.NUM_CLASSES))

if outdir == None:
    outdir = cfg.OUTPUT_DIR
else:
    pathlib.Path(outdir).mkdir(exist_ok=True)

print("Evaluation output directory: " + outdir)

cfg.DATASETS.TEST = (dataset,)
model = build_model(cfg)


## Test
#

if use_gt_bboxes:
    # returns a list of dicts. Every entry in the list corresponds to one sample, represented by a dict.
    dataset_dicts = detectron2.data.get_detection_dataset_dicts(dataset)

    # add proposal boxes
    for i, s in enumerate(dataset_dicts):
        s["proposal_boxes"] = np.array([ ann["bbox"] for ann in dataset_dicts[i]["annotations"] ]) # np.array([[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ...], ...]) # kx4 matrix for k proposed bounding boxes
        s["proposal_objectness_logits"] = np.full((s["proposal_boxes"].shape[0],), 10) # logit of 10 is 99.999...%
        s["proposal_bbox_mode"] = detectron2.structures.BoxMode.XYWH_ABS # 1 # (x0, y0, w, h) in absolute floating points coordinates 
    
    print("Proposal boxes added.")

    model.proposal_generator = None # deactivate such that precomputed proposals are used
    print("Region proposal deactivated, ground truth bounding boxes are used.")

    val_loader = build_detection_test_loader(dataset_dicts, mapper=DatasetMapper(is_train=False, augmentations=[], image_format= cfg.INPUT.FORMAT, precomputed_proposal_topk=500))
else:
    val_loader = build_detection_test_loader(cfg, dataset)

if last_only:
    DetectionCheckpointer(model).load(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
    #DetectionCheckpointer(model).load(os.path.join(cfg.OUTPUT_DIR, "model_0099999.pth"))

    evaluator = COCOEvaluator(dataset, ("bbox",), False, output_dir=outdir)
    result = inference_on_dataset(model, val_loader, evaluator)
    print_csv_format(result)

    with open(outdir + "/evaluation_" + dataset + ".json", "w") as outfile:
            json.dump(result, outfile)
else:
    files = glob.glob(cfg.OUTPUT_DIR + "/model_*.pth")

    # remove files for which evaluation has already been done
    already_evaluated = glob.glob(outdir + "/evaluation_*")
    files = [f for f in files if outdir + "/evaluation_" + f.strip(cfg.OUTPUT_DIR).strip("/model_").strip(".pth") + ".json" not in already_evaluated]

    for i, f in enumerate(files):
        DetectionCheckpointer(model).load(f)

        evaluator = COCOEvaluator(dataset, ("bbox",), False, output_dir=outdir)
        result = inference_on_dataset(model, val_loader, evaluator)
        print_csv_format(result)

        with open(outdir + "/evaluation_" + f.strip(cfg.OUTPUT_DIR).strip("/model_").strip(".pth") + ".json", "w") as outfile:
            json.dump(result, outfile)
