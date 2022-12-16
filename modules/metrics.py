from torchmetrics import Metric
from torchdrug.metrics import metric
import pdb
import torch

class f1_max(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("pred", default=[], dist_reduce_fx="cat") # [batch_size, num_classes]
        self.add_state("target", default=[], dist_reduce_fx="cat") # [batch_size, num_classes]
    
    def update(self, pred, target):
        assert len(pred) == len(target), "pred and target must have the same length"
        self.pred.append(pred) # finally a list contain [batch_size, num_classes]*num_steps
        self.target.append(target)
    
    def compute(self):
        # The torch_metrics will reduce a large list of [batch_size, num_classes] of length num_steps*num_gpus
        # finally we get a [train_dataset_size, num_classes] tensor
        # pdb.set_trace()
        if isinstance(self.pred, list):
            # pdb.set_trace()
            self.pred = torch.cat(self.pred, dim=0)
            self.target = torch.cat(self.target, dim=0)
        num_class = self.pred.shape[-1]
        self.pred = self.pred.reshape(-1, num_class)
        self.target = self.target.reshape(-1, num_class)
        assert len(self.pred.shape) == 2, "pred must be a 2D tensor of [Batch, NClass]"
        assert len(self.target.shape) == 2, "target must be a 2D tensor of [Batch, NClass]"
        f1_max = metric.f1_max(self.pred, self.target)
        
        return f1_max


    
    