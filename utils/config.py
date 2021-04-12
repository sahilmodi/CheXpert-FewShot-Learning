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
_C.SOLVER.MIXUP_ALPHA = 0.0

_C.TEACHER = CfgNode()
_C.TEACHER.EPOCHS = 15
_C.TEACHER.BETA_L = 0.0
_C.TEACHER.BETA_U = 0.0
_C.TEACHER.BETA_C = 0.15

_C.STUDENT = CfgNode()
_C.STUDENT.EPOCHS = 15
_C.STUDENT.BETA_L = 0.0
_C.STUDENT.BETA_U = 0.8
_C.STUDENT.BETA_C = 0.15