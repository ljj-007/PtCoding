from .utils import *
from .model.model_process import TestProcessor

class Tester():
    def __init__(self, args, data, model) -> None:
        self.args = args
        self.data = data
        self.model = model
        self.test_processor = TestProcessor(self.args, self.data)

    def run_epoch(self):
        return self.test_processor.process_epoch(self.model)