
import torch

from utils.logger import get_logger


logger = get_logger("Device")


def get_device() -> torch.device:
    """
    Return the best available device.
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        logger.info("CUDA Available")

        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")

        logger.info(
            f"CUDA Version : {torch.version.cuda}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        logger.info(
            f"GPU Memory : {total_memory:.2f} GB"
        )

        logger.info(
            f"Device Count : {torch.cuda.device_count()}"
        )

        logger.info(
            f"BF16 Supported : {torch.cuda.is_bf16_supported()}"
        )

    else:

        device = torch.device("cpu")

        logger.warning("CUDA Not Available")

        logger.warning("Using CPU")

    return device


def clear_cache():

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def synchronize():

    if torch.cuda.is_available():

        torch.cuda.synchronize()
