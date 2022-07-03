"""
Author:  Cax
File:    config
Project: GAN
Time:    2022/6/30
Des:     Default parameter settings.
"""
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.PATH = 'datasets'
_C.DATASETS.MEAN = (0.5,)
_C.DATASETS.STA = (0.5,)
_C.DATASETS.IMG = CN()
_C.DATASETS.IMG.DIM = 28
_C.DATASETS.LAB = CN()
_C.DATASETS.LAB.NUM = 10

# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.SHUFFLE = True

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
_C.MODEL.GENERATOR = CN()
_C.MODEL.GENERATOR.LABEL_EMB = 10
_C.MODEL.GENERATOR.FAKE_DIM = 100

_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.LABEL_EMB = 10
# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 5
_C.SOLVER.LOSS = 'BCELoss'
_C.SOLVER.GENERATOR = CN()
_C.SOLVER.GENERATOR.OPTIM = 'Adam'
_C.SOLVER.GENERATOR.LR = 1e-4
_C.SOLVER.DISCRIMINATOR = CN()
_C.SOLVER.DISCRIMINATOR.OPTIM = 'Adam'
_C.SOLVER.DISCRIMINATOR.LR = 1e-4

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
_C.OUTPUT_DIR = "."

