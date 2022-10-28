import torch
import torch.nn as nn


from data_preprocess import load_imdb
from torch.utils.data import DataLoader
from utils import set_seed
from torchvision.models import resnet34


