# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

_C.OUTPUT_ROOT_DIR = ''

_C.DATA = CfgNode()
_C.DATA.PATH = '/home/smodi9/CheXpert-v1.0-small' 
_C.DATA.BATCH_SIZE = 100
_C.DATA.LABELED_SIZE = 1_000
_C.DATA.UNLABELED_SIZE = 15_000
