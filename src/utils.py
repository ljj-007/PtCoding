import os
import shutil
import logging
import random
import sys
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import pandas as pd
import datetime
from transformers import BertTokenizer, BertModel

from tqdm import tqdm


def get_datetime():
    """ Get Current Time """
    return datetime.datetime.now().strftime("%Y-%M-%d-%H-%M-%S")

def set_seeds(seed):
    """ Set Random Seeds """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "main":
    pass