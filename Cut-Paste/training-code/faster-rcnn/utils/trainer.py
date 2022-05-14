import numpy as np
import logging
import detectron2

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.modeling import build_model


# subclass default trainer and implent its build_evaluator function to enable periodic evaluation during the training
class COCOTrainer(DefaultTrainer):
  
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            COCO DatasetEvaluator for "bbox" mode.
        """

        return detectron2.evaluation.COCOEvaluator(dataset_name, ("bbox",), False, output_dir= cfg.OUTPUT_DIR)

class COCOGroundTruthBoxesTrainer(DefaultTrainer):
    """
    Custum trainer for the use of pre-specified bounding boxes instead of a region proposal network.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            COCO DatasetEvaluator for "bbox" mode.
        """

        return detectron2.evaluation.COCOEvaluator(dataset_name, ("bbox",), False, output_dir= cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        custom dataloader to provide model with ground truth bounding boxes
        """

        # returns a list of dicts. Every entry in the list corresponds to one sample, represented by a dict.
        dataset_dicts = detectron2.data.get_detection_dataset_dicts(cfg.DATASETS.TRAIN[0])

        # add proposal boxes
        for i, s in enumerate(dataset_dicts):
            s["proposal_boxes"] = np.array([ ann["bbox"] for ann in dataset_dicts[i]["annotations"] ]) # np.array([[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ...], ...]) # kx4 matrix for k proposed bounding boxes
            s["proposal_objectness_logits"] = np.full((s["proposal_boxes"].shape[0],), 10) # logit of 10 is 99.999...%
            s["proposal_bbox_mode"] = detectron2.structures.BoxMode.XYWH_ABS # 1 # (x0, y0, w, h) in absolute floating points coordinates

        print("Proposal boxes added.")

        return build_detection_train_loader(dataset_dicts, mapper=DatasetMapper(is_train=True, augmentations=[], image_format=cfg.INPUT.FORMAT, precomputed_proposal_topk=500),
                                            total_batch_size=cfg.SOLVER.IMS_PER_BATCH, aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING, num_workers=cfg.DATALOADER.NUM_WORKERS)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Custom dataloader to provide model with ground truth bounding boxes
        """
        # returns a list of dicts. Every entry in the list corresponds to one sample, represented by a dict.
        dataset_dicts = detectron2.data.get_detection_dataset_dicts(dataset_name)

        # add proposal boxes
        for i, s in enumerate(dataset_dicts):
            s["proposal_boxes"] = np.array([ ann["bbox"] for ann in dataset_dicts[i]["annotations"] ]) # np.array([[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ...], ...]) # kx4 matrix for k proposed bounding boxes
            s["proposal_objectness_logits"] = np.full((s["proposal_boxes"].shape[0],), 10) # logit of 10 is 99.999...%
            s["proposal_bbox_mode"] = detectron2.structures.BoxMode.XYWH_ABS # 1 # (x0, y0, w, h) in absolute floating points coordinates

        print("Proposal boxes for test data added.")

        return build_detection_test_loader(dataset_dicts, mapper=DatasetMapper(is_train=False, augmentations=[], image_format= cfg.INPUT.FORMAT, precomputed_proposal_topk=500))

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        Custom model builder, which deactivates region proposal such that provided ground truth bboxes are used as proposals instead.
        """
        model = build_model(cfg)
        model.proposal_generator = None

        print("Region proposal deactivated, ground truth bounding boxes are used.")

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model