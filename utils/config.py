# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

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
_C.SOLVER.NUM_EPOCHS = 20
_C.SOLVER.VAL_INTERVAL = 5