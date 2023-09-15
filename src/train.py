from .utils import *
from .model.model_process import TrainProcessor, ValidProcessor

class Trainer():
    def __init__(self, args, data, model, optimizer) -> None:
        self.args = args
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.train_processor = TrainProcessor(args, data)
        self.valid_processor = ValidProcessor(args, data)

    def run_epoch(self):
        total_loss = self.train_processor.process_epoch(self.model, self.optimizer)
        res = self.valid_processor.process_epoch(self.model)
        return total_loss, res