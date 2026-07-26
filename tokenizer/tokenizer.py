
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from configs.config import Config
from utils.logger import get_logger


logger = get_logger("Tokenizer")


class QualcTokenizer:

    def __init__(self):

        self.tokenizer = Tokenizer(
            BPE(
                unk_token=Config.UNK_TOKEN
            )
        )

        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])

        self.tokenizer.pre_tokenizer = ByteLevel()

        self.tokenizer.decoder = ByteLevelDecoder()

        self.trainer = BpeTrainer(
            vocab_size=Config.VOCAB_SIZE,
            min_frequency=2,
            special_tokens=[
                Config.PAD_TOKEN,
                Config.BOS_TOKEN,
                Config.EOS_TOKEN,
                Config.UNK_TOKEN,
            ],
        )

    def train(self, source, length=None):

        logger.info("Tokenizer Training Started")

        if isinstance(source, (str, list, tuple)):
            self.tokenizer.train(
                files=source,
                trainer=self.trainer,
            )
        else:
            self.tokenizer.train_from_iterator(
                iterator=source,
                trainer=self.trainer,
                length=length,
            )

        logger.info("Tokenizer Training Completed")

    def save(self):

        Path(Config.TOKENIZER_DIR).mkdir(
            parents=True,
            exist_ok=True,
        )

        save_path = Path(
            Config.TOKENIZER_DIR
        ) / "qualc_tokenizer.json"

        self.tokenizer.save(str(save_path))

        logger.info(f"Tokenizer Saved : {save_path}")

    def load(self):

        tokenizer_path = Path(
            Config.TOKENIZER_DIR
        ) / "qualc_tokenizer.json"

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        logger.info("Tokenizer Loaded")

    def encode(self, text):

        return self.tokenizer.encode(text)

    def decode(self, ids):

        return self.tokenizer.decode(ids)

    def vocab_size(self):

        return self.tokenizer.get_vocab_size()
