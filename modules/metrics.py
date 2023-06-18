import pdb

import torch
from torchdrug.metrics import metric
from torchmetrics import Metric


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

class area_under_prc(Metric):
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
        auprc_micro = metric.area_under_prc(self.pred, self.target)
        
        return auprc_micro
    
class variadic_area_under_prc(Metric):
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
        auprc_macro = metric.variadic_area_under_prc(self.pred, self.target, dim=0).mean()
        
        return auprc_macro

class classification_metric(Metric):
    def __init__(self, metric_name):
        super().__init__()
        self.add_state("pred", default=[], dist_reduce_fx="cat") # [batch_size, num_classes]
        self.add_state("target", default=[], dist_reduce_fx="cat") # [batch_size, num_classes] or [batch_size]
        self.metric_name = metric_name
    
    def update(self, pred, target):
        assert len(pred) == len(target), "pred and target must have the same length"
        self.pred.append(pred)
        self.target.append(target)

    def compute(self):
        if isinstance(self.pred, list):
            self.pred = torch.cat(self.pred, dim=0)
            self.target = torch.cat(self.target, dim=0)
        num_class = self.pred.shape[-1]
        # print(f"pred shape is {self.pred.shape}")
        # print(f"target shape is {self.target.shape}")
        # print(self.target.shape)
        self.pred = self.pred.reshape(-1, num_class)
        # self.target = self.target.reshape(-1, num_class)
        assert len(self.pred.shape) == 2, "pred must be a 2D tensor of [Batch, NClass]"
        # assert len(self.target.shape) == 2, "target must be a 2D tensor of [Batch, NClass]"

        metric_func = getattr(metric, self.metric_name)
        if self.metric_name == "variadic_area_under_prc" or self.metric_name == "variadic_area_under_roc":
            metric_value = metric_func(self.pred, self.target.long(), dim=0).mean()
        elif self.metric_name == "area_under_prc" or self.metric_name == "area_under_roc":
            metric_value = metric_func(self.pred.flatten(), self.target.long().flatten())
        else:
            metric_value = metric_func(self.pred, self.target)

        return metric_value
    
    