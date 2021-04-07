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
_C.DATA.NUM_WORKERS = 8
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CfgNode()
_C.INPUT.IMAGE_CHANNEL = 3
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY =  0
_C.SOLVER.SCHEDULER_STEP_SIZE = 15
_C.SOLVER.NUM_ITERS = 20
_C.SOLVER.VAL_INTERVAL = 5
_C.SOLVER.MIXUP_ALPHA = 0