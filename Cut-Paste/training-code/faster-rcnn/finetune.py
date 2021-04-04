import os
import argparse
import datetime
import pathlib
import cv2
import torch
import detectron2

from detectron2 import model_zoo
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.datasets import register_coco_instances

from utils.trainer import COCOTrainer, COCOGroundTruthBoxesTrainer


## Initialization
#

#{date:%Y-%m-%d_%H%M}".format(date=datetime.datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="output/2021-02-12_2309", help="Path to output folder (will be created if it does not exist).")
parser.add_argument("--resume", action='store_true', default=True, help="Set to resume from last checkpoint in outdir.")

parser.add_argument("--annotations", type=str, default="/home/dimitar/train_annotations.json", help="Path to COCO-style annotations file.")
parser.add_argument("--imagedir", type=str, default="/home/mengmi/Projects/Proj_context2/Datasets/MSCOCO/trainColor_oriimg",  help="Path to images folder w.r.t. which filenames are specified in the annotations.")
parser.add_argument("--num_classes", type=int, default=55, help="Number of classes.")

parser.add_argument("--test_annotations", type=str, default='/home/dimitar/test_annotations.json', help="Path to COCO-style annotations file for model evaluation.")
parser.add_argument("--test_imagedir", type=str, default='/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expH', help="Path to images folder w.r.t. which filenames are specified in the annotations for model evaluation.")
parser.add_argument("--test_frequency", type=int, default=50000, help="Evaluate model on test data every __ iterations.")

parser.add_argument("--iters", type=int, default=500000, help="Number of iterations to train.")
parser.add_argument("--save_frequency", type=int, default=50000, help="Save model checkpoint every __ iterations.")
args = parser.parse_args()


# Register datasets
register_coco_instances("train", {}, args.annotations, args.imagedir)
register_coco_instances("test", {}, args.test_annotations, args.test_imagedir)

# Load and configure model
cfg = model_zoo.get_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", trained=True)

cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST =  ("test",)

cfg.OUTPUT_DIR = args.outdir

cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.CHECKPOINT_PERIOD = args.save_frequency
cfg.SOLVER.MAX_ITER = args.iters # Note that when traininig is resumed the iteration count will resume as well, so increase the number of iterations to train further. 
cfg.TEST.EVAL_PERIOD = args.test_frequency
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 55

# save config
pathlib.Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
with open(cfg.OUTPUT_DIR + "/model.yaml", "w") as f:
    f.write(cfg.dump())


## Train
#

trainer = COCOGroundTruthBoxesTrainer(cfg) # COCOTrainer(cfg) can be used instead to activate region proposal network instead of using gt bboxes
trainer.resume_or_load(resume=args.resume)
trainer.train()
