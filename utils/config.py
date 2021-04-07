# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

_C.OUTPUT_ROOT_DIR = ''

_C.DATA = CfgNode()
_C.DATA.PATH = './data/'
_C.DATA.BATCH_SIZE = 100
_C.DATA.TRAIN_SIZE = 1_000_000
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.WEIGHT_DECAY =  1e-4
_C.SOLVER.SCHEDULER_STEP_SIZE = 5
_C.SOLVER.NUM_EPOCHS = 15
_C.SOLVER.MIXUP_ALPHA = 0.0