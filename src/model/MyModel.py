from ..utils import *

class MyModel(nn.Module):
    def __init__(self, args) -> None:
        super(MyModel, self).__init__()
        self.args = args
        self.loss_func = ...

    def loss(self, inputs, targets):
        """ Training """
        outputs = ...
        loss = self.loss_func(outputs, targets)
        return loss

    def predict(self, inputs):
        """ Validing and Testing """
        outputs = ...
        return outputs