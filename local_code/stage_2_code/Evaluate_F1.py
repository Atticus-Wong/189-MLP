'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score
import numpy as np


class Evaluate_F1(evaluate):
    data = None
    
    def evaluate(self):
        return f1_score(self.data['true_y'], self.data['pred_y'], average="weighted")
        