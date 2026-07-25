
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:

    # ==========================================================
    # Project
    # ==========================================================

    PROJECT_NAME = "Qualc-Pipeline"
    MODEL_NAME = "Qualc-LM"
    MODEL_TYPE = "decoder"

    VERSION = "0.1.0"

    SEED = 42

    # ==========================================================
    # Directories
    # ==========================================================

    ROOT_DIR = Path(__file__).resolve().parent.parent

    OUTPUT_DIR = ROOT_DIR / "outputs"

    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

    LOG_DIR = OUTPUT_DIR / "logs"

    TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"

    # ==========================================================
    # Dataset
    # ==========================================================

    DATASET_NAME = ""

    DATASET_CONFIG = None

    DATASET_SPLIT = "train"

    TRAIN_SPLIT = 0.95

    VALID_SPLIT = 0.05

    SHUFFLE_DATASET = True

    NUM_WORKERS = 2

    # ==========================================================
    # Tokenizer
    # ==========================================================

    VOCAB_SIZE = 50000

    MIN_FREQUENCY = 2

    MAX_SEQUENCE_LENGTH = 2048

    PAD_TOKEN = "<pad>"

    BOS_TOKEN = "<bos>"

    EOS_TOKEN = "<eos>"

    UNK_TOKEN = "<unk>"

    SPECIAL_TOKENS = [
        PAD_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        UNK_TOKEN,
    ]

    # ==========================================================
    # Model
    # ==========================================================

    HIDDEN_SIZE = 1024

    NUM_LAYERS = 24

    NUM_HEADS = 16

    NUM_KV_HEADS = 4

    INTERMEDIATE_SIZE = 4096

    MAX_POSITION_EMBEDDINGS = 2048

    DROPOUT = 0.0

    ATTENTION_DROPOUT = 0.0

    RMS_NORM_EPS = 1e-6

    ROPE_THETA = 10000.0

    ROPE_SCALING = None

    INIT_STD = 0.02

    ACTIVATION = "swiglu"

    ATTENTION_BIAS = False

    MLP_BIAS = False

    USE_ROPE = True

    USE_GQA = True

    USE_SWIGLU = True

    USE_KV_CACHE = True

    USE_FLASH_ATTENTION = True

    USE_SDPA = True

    USE_MIXED_PRECISION = True

    USE_GRADIENT_SCALER = True

    GRADIENT_CHECKPOINTING = False

    USE_TORCH_COMPILE = False

    # ==========================================================
    # Training
    # ==========================================================

    MODE = "pretrain"

    BATCH_SIZE = 4

    GRADIENT_ACCUMULATION = 8

    EPOCHS = 1

    LEARNING_RATE = 3e-4

    WEIGHT_DECAY = 0.1

    WARMUP_STEPS = 1000

    MAX_GRAD_NORM = 1.0

    # ==========================================================
    # Precision
    # ==========================================================

    USE_FP16 = False

    USE_BF16 = torch.cuda.is_available()

    USE_TF32 = torch.cuda.is_available()

    # ==========================================================
    # DataLoader
    # ==========================================================

    PIN_MEMORY = torch.cuda.is_available()

    PERSISTENT_WORKERS = True

    PREFETCH_FACTOR = 2

    # ==========================================================
    # Validation
    # ==========================================================

    EVAL_EVERY = 1000

    SAVE_BEST_MODEL = True

    EARLY_STOPPING = False

    EARLY_STOPPING_PATIENCE = 5

    # ==========================================================
    # Checkpoint
    # ==========================================================

    SAVE_EVERY = 1000

    KEEP_LAST = 5

    RESUME_TRAINING = True


    CHECKPOINT_INTERVAL = 1000

    KEEP_LAST_CHECKPOINTS = 5

    SAVE_LATEST = True

    SAVE_BEST = True

    SAVE_FINAL = True

    AUTO_RESUME = True

    CHECKPOINT_EXTENSION = ".pt"

    CHECKPOINT_PREFIX = "step_"

    LATEST_CHECKPOINT_NAME = "latest.pt"

    BEST_CHECKPOINT_NAME = "best.pt"

    FINAL_CHECKPOINT_NAME = "final.pt"

    CHECKPOINT_METADATA = True


    # ==========================================================
    # Device
    # ==========================================================

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ==========================================================
    # Logging
    # ==========================================================

    LOG_EVERY = 10

    LOG_LEVEL = "INFO"

    # ==========================================================
    # Evaluation
    # ==========================================================

    COMPUTE_PERPLEXITY = True

    COMPUTE_VALIDATION_LOSS = True

    GENERATE_SAMPLE = True

    SAMPLE_PROMPT = "Hello"

    # ==========================================================
    # Inference
    # ==========================================================

    MAX_NEW_TOKENS = 256

    TEMPERATURE = 0.8

    TOP_P = 0.95

    TOP_K = 50

    REPETITION_PENALTY = 1.1

    DO_SAMPLE = True

    USE_CACHE = True

    RETURN_ATTENTION = False

    RETURN_HIDDEN_STATES = False

    # ==========================================================
    # Utility Methods
    # ==========================================================

    @classmethod
    def create_directories(cls):

        directories = [
            cls.OUTPUT_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.TOKENIZER_DIR,
        ]

        for directory in directories:

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    @classmethod
    def get_dtype(cls):

        if (
            cls.USE_BF16
            and torch.cuda.is_available()
        ):
            return torch.bfloat16

        if (
            cls.USE_FP16
            and torch.cuda.is_available()
        ):
            return torch.float16

        return torch.float32

    @classmethod
    def validate(cls):

        assert (
            cls.HIDDEN_SIZE %
            cls.NUM_HEADS == 0
        ), (
            "HIDDEN_SIZE must be divisible "
            "by NUM_HEADS."
        )

        assert (
            cls.NUM_HEADS %
            cls.NUM_KV_HEADS == 0
        ), (
            "NUM_HEADS must be divisible "
            "by NUM_KV_HEADS."
        )

        assert (
            cls.TRAIN_SPLIT +
            cls.VALID_SPLIT
            == 1.0
        ), (
            "TRAIN_SPLIT + VALID_SPLIT "
            "must equal 1.0."
        )

        assert (
            cls.VOCAB_SIZE > 0
        ), "VOCAB_SIZE must be positive."

        assert (
            cls.MAX_SEQUENCE_LENGTH > 0
        ), (
            "MAX_SEQUENCE_LENGTH "
            "must be positive."
        )

        assert (
            cls.MAX_POSITION_EMBEDDINGS >=
            cls.MAX_SEQUENCE_LENGTH
        ), (
            "MAX_POSITION_EMBEDDINGS "
            "must be greater than or equal "
            "to MAX_SEQUENCE_LENGTH."
        )

        assert (
            cls.BATCH_SIZE > 0
        ), (
            "BATCH_SIZE must be positive."
        )

        assert (
            cls.LEARNING_RATE > 0
        ), (
            "LEARNING_RATE must be positive."
        )

    @classmethod
    def initialize(cls):

        cls.create_directories()

        cls.validate()
        
