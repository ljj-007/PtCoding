from ..utils import *

class MyData():
    def __init__(self, args) -> None:
        self.args = args
        self.train_data = list()
        self.test_data = list()
        self.load_data()

    def load_data(self):
        """ Load data and pre-solve it, before inputing to class [Dataset] """
        self.train_data = ...
        self.test_data = ...