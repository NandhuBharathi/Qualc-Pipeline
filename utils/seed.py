
import os
import random

import numpy as np
import torch

from utils.logger import get_logger


logger = get_logger("Seed")


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    logger.info(f"Random Seed : {seed}")
