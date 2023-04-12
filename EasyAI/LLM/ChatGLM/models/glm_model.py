# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/1 9:20 PM
# @File: glm_model
# @Email: mlshenkai@163.com
""" Pytorch ChatGlm Model """
import math
import copy
import os
import warnings
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable

from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast,  BaseModelOutputWithPastAndCrossAttentions

from transformers.modeling_utils import PreTrainedModel
from loguru import logger
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig

from ..config import ChatGLMConfig

torch._C._jit_set_profiling_mode(False)

