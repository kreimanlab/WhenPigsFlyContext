import unittest
import sys

import torch

sys.path.append("../")
from utils.metrics import AccuracyLogger

class AccuracyLoggerTest(unittest.TestCase):

    def test_confusionmatrix(self):
        idx2label_dict = {0: "class1", 1: "class2", 2: "class3"}

        # sample output for batchsize 2 and 3 classes (logits)
        output = torch.tensor([[1.3, 0.2, -0.3],
                               [0.1, 2.4,  1.4]])
        
        # sample ground truth: for first batch, class 0 is true, for 2nd batch, class 2 is true
        groundtruth = torch.tensor([0, 2])

        accuracy_logger = AccuracyLogger(idx2label_dict)
        accuracy_logger.update(output, groundtruth)

        self.assertEqual(accuracy_logger.confusionmatrix[0,0], 1)
        self.assertEqual(accuracy_logger.confusionmatrix[0,1], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[0,2], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[1,0], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[1,1], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[1,2], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[2,0], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[2,1], 1)
        self.assertEqual(accuracy_logger.confusionmatrix[2,2], 0)

        # add more samples from a next hypothetical batch
        output = torch.tensor([[1.3, 0.2, -0.3],
                               [0.1, 2.4, 1.4 ],
                               [0.0, 1.1, 1.8 ]])
        groundtruth = torch.tensor([0, 2, 2])
        accuracy_logger.update(output, groundtruth)

        self.assertEqual(accuracy_logger.confusionmatrix[0,0], 2)
        self.assertEqual(accuracy_logger.confusionmatrix[0,1], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[0,2], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[1,0], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[1,1], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[1,2], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[2,0], 0)
        self.assertEqual(accuracy_logger.confusionmatrix[2,1], 2)
        self.assertEqual(accuracy_logger.confusionmatrix[2,2], 1)


    def test_state_dict(self):
        
        idx2label_dict = {0: "class1", 1: "class2", 2: "class3"}
        accuracy_logger = AccuracyLogger(idx2label_dict)

        # add some data to the logger
        output = torch.tensor([[1.3, 0.2, -0.3],
                               [0.1, 2.4,  1.4]])
        groundtruth = torch.tensor([0, 2])
        accuracy_logger.update(output, groundtruth)

        confusionmatrix_before = accuracy_logger.confusionmatrix
        distributions_before = accuracy_logger.distributions

        # get state_dict
        state_dict = accuracy_logger.state_dict()

        # initialize logger from state_dict
        restored_accuracy_logger = AccuracyLogger.from_state_dict(state_dict)

        confusionmatrix_after = restored_accuracy_logger.confusionmatrix
        distributions_after = restored_accuracy_logger.distributions

        self.assertTrue(torch.equal(confusionmatrix_before, confusionmatrix_after))
        self.assertTrue(torch.equal(distributions_before, distributions_after))