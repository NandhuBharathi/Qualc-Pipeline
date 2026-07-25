
from datasets import load_dataset

from configs.config import Config
from utils.logger import get_logger


logger = get_logger("Dataset")


class DatasetLoader:

    def __init__(self):

        self.dataset_name = Config.DATASET_NAME
        self.dataset_config = Config.DATASET_CONFIG
        self.dataset_split = Config.DATASET_SPLIT

    def load(self):

        logger.info(f"Loading Dataset : {self.dataset_name}")

        if self.dataset_config:

            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.dataset_split
            )

        else:

            dataset = load_dataset(
                self.dataset_name,
                split=self.dataset_split
            )

        logger.info(f"Total Samples : {len(dataset)}")

        return dataset

    def train_validation_split(self, dataset):

        split = dataset.train_test_split(
            test_size=Config.VALID_SPLIT,
            shuffle=Config.SHUFFLE_DATASET,
            seed=Config.SEED
        )

        train_dataset = split["train"]
        validation_dataset = split["test"]

        logger.info(
            f"Training Samples : {len(train_dataset)}"
        )

        logger.info(
            f"Validation Samples : {len(validation_dataset)}"
        )

        return train_dataset, validation_dataset
