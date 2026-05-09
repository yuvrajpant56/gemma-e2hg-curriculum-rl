import os
MODEL_NAME = "google/gemma-4-E2B-it"


TRAIN_DATASET_PATHS = [
    "local_data/countdown_n2_train.parquet",
    "local_data/countdown_n3_train.parquet",
    "local_data/countdown_n4_train.parquet",
    "local_data/countdown_n5_train.parquet",
]

EVAL_DATASET_PATHS = {
    "n2": "local_data/countdown_n2_test.parquet",
    "n3": "local_data/countdown_n3_test.parquet",
    "n4": "local_data/countdown_n4_test.parquet",
    "n5": "local_data/countdown_n5_test.parquet",
}


MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 1.0
DO_SAMPLE = True
DEVICE_MAP = "auto"
TORCH_DTYPE = "auto"

HF_TOKEN = os.environ.get("HF_TOKEN", "")


USE_LORA =True
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

