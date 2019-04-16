# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.
"""

from fcn.config import cfg
from gt_data_layer.minibatch import get_minibatch
import numpy as np
#from utils.voxelizer import Voxelizer

class GtDataLayer(object):

    def __init__(self):

        return 1