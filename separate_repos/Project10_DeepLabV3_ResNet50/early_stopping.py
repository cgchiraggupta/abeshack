import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, path="checkpoints/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"⏸ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f"✅ Saved new best model to {self.path} (EarlyStopping)")