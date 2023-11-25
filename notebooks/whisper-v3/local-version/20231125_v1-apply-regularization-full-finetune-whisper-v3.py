#!/usr/bin/env python
# coding: utf-8

# # Data prep

# In[1]:


import re
import requests
import string
import os
import subprocess
import multiprocessing as mp
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
from itertools import compress, chain

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import datasets
import torch
import jiwer
import numpy as np
import pandas as pd
import tensorflow as tf   
from IPython.display import Audio as audio_display
from deepcut import tokenize  # Consume too much memory when using with CUDA
# from pythainlp.tokenize import word_tokenize as tokenize
from pythainlp.util import normalize
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, pipeline, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, load_from_disk, interleave_datasets
from datasets.features import Audio

tf.keras.utils.disable_interactive_logging()


# In[2]:


def print_gpu_info():
    gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
    if gpu_info.stdout.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info.stdout)
        
print_gpu_info()


# *The common voice data can be downloaded from this command*  
# ```
# !wget https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-15.0-2023-09-08/cv-corpus-15.0-2023-09-08-th.tar.gz\?X-Goog-Algorithm\=GOOG4-RSA-SHA256\&X-Goog-Credential\=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20231120%2Fauto%2Fstorage%2Fgoog4_request\&X-Goog-Date\=20231120T211937Z\&X-Goog-Expires\=43200\&X-Goog-SignedHeaders\=host\&X-Goog-Signature\=7b63c1ccdb27c7a2f2b1b5e59422ab38668543f242283238b92d39552aa12a2686ba413b29107e71c8fa75d850decf8d5f9e1f5c0f6b72da42154cf478ebe296f8445d1744267a3ad40391433517c9ad8735b26cfe5c53e777feffac2a71d54ee7ce47cb1c580449340a84d066271a57a2beba416de0d7e897ad7bd99f13e68e0d8a1a2cc1c2dbf2341740fd167e1d6572d84b23c9daee4139dd35cc8f827db052a05021ca1c25549baa18c823ed1c25347cd10972451718ac13c73b656bbc69134ebbcce7206ad38c6e3611ac59881e8a630abbdf7390b689bb74d7fe35cb80366742d76cf5a6eb462e6da408dd2bb05a97cd8b89a4110479d62f9dc6f84c4e
# ```

# In[3]:


# Device config
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Data config
TRAIN_SET_DIST = (0.1, 0.15, 0.75)
SEED = 4242
SAMPLING = 1  # sampling rate
AUDIO_SAMPLING_RATE = 16_000
MODEL_PATH_OR_URL = "20231125-model-backup/checkpoint-800"
NUM_SHARDS = 100

EVAL_MAX_N_FILES = None
TRAIN_MAX_N_FILES = None
TEST_MAX_N_FILES = None

# DATA processing config
NUM_PROC = 4
WRITER_BATCH_SIZE = 128
DATA_PROCESS_BATCH_SIZE = 128

config = {
    "num_proc": NUM_PROC, 
    "keep_in_memory": False, 
    "load_from_cache_file": True, 
    "writer_batch_size": WRITER_BATCH_SIZE,
    "load_from_cache_file": True,
    "batch_size": DATA_PROCESS_BATCH_SIZE,
    "batched": True,
}

# model config
LEARNING_RATE = 0.25e-7
WARMUP_STEPS = 100
MAX_STEPS = 2000
SAVE_STEPS = 400
EVAL_STEPS = 20
LOGGING_STEPS = 10
OPTIMIZER = "adamw_bnb_8bit"

MAX_LABEL_LENGTH = 448
MAX_INPUT_LENGTH = 30
GENERATION_MAX_LENGTH = 225
CHUNK_LENGTH = 30
NUM_BEAMS = 1
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
N = 4

# model regularization config
DROPOUT = 0.05
APPLY_SPEC_AUGMENT = True
MASK_FEATURE_LENGTH = 10
MASK_FEATURE_PROB = 0.05
MASK_FEATURE_MIN_MASKS = 1
MASK_TIME_PROB = 0.05
MASK_TIME_MIN_MASKS = 2
MASK_TIME_LENGTH = 10

# resource config
MODEL_CHECKPOINT_DIR = Path.cwd() / "fine-tune-whisper-large-v3-checkpoints"
DATASET_CACHE_DIR = Path.cwd() / "dataset-cache"
FINAL_MODEL_OUTPUT_DIR = Path.cwd() / "fine-tune-model"
FINAL_PROC_OUTPUT_DIR = Path.cwd() / "fine-tune-proc"

PRJ_ROOT = Path.cwd().parents[2]
DATA_PATH = PRJ_ROOT / "notebooks" / "whisper-v3" / "data"
COMMON_VOICE_PATH = DATA_PATH /  "cv-corpus-15" / "th"
AUDIO_BASE_PATH = COMMON_VOICE_PATH / "clips"

OPP_DAY_PATH = DATA_PATH / "Opp Day SRTs"
OPP_DAY_AUDIO_BASE_PATH = OPP_DAY_PATH / "chunks"

GOWAJEE_PATH = DATA_PATH / "gowajee"
GOWAJEE_AUDIO_BASE_PATH = GOWAJEE_PATH / "audios"

TEST_PATH = DATA_PATH / "stt-dataset" / "test"
TEST_AUDIO_BASE_PATH = TEST_PATH / "[TH] Oppday Q2_2023 IP บมจ. อินเตอร์ ฟาร์มา"


# In[4]:


def remove_punct(s: str) -> str:
    s = re.sub(rf"[{re.escape(string.punctuation)}]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = normalize(s)
    return s

remove_punct("ไหน?ลองซิ! ...")


# In[36]:


gow_train = (GOWAJEE_PATH / "train" / "text").read_text().split("\n")
gow_dev = (GOWAJEE_PATH / "dev" / "text").read_text().split("\n")
gow_test = (GOWAJEE_PATH / "test" / "text").read_text().split("\n")

gow_data = chain(gow_train, gow_dev, gow_test)
gow_data = list(map(lambda s: ((ls := s.split(" "))[0] + ".wav", "".join(ls[1:])), gow_data))


# In[49]:


# Common Voice Dataset
common_voice_train = (
    pd.concat([
        pd.read_csv(str(COMMON_VOICE_PATH / "train.tsv"), sep="\t"),
    ])
    .assign(full_path=lambda df: AUDIO_BASE_PATH / df.path)
    .assign(sentence=lambda df: df.sentence.map(remove_punct))
)

common_voice_eval = (
    pd.concat([
        pd.read_csv(str(COMMON_VOICE_PATH / "dev.tsv"), sep="\t"),
    ])
    .assign(full_path=lambda df: AUDIO_BASE_PATH / df.path)
    .assign(sentence=lambda df: df.sentence.map(remove_punct))
)

common_voice_test = (
    pd.concat([
        pd.read_csv(str(COMMON_VOICE_PATH / "test.tsv"), sep="\t"),
    ])
    .assign(full_path=lambda df: AUDIO_BASE_PATH / df.path)
    .assign(sentence=lambda df: df.sentence.map(remove_punct))
).iloc[:TEST_MAX_N_FILES]

# SET Opp Day dataset
amm_opp_data_df = (
    pd.concat([
        pd.read_csv(str(OPP_DAY_PATH / "train_metadata.tsv"), sep="\t"),
    ])
)
amm_opp_data_df = (
    amm_opp_data_df[amm_opp_data_df.transcript.str.len() > 2]
    .assign(full_path=lambda df: OPP_DAY_AUDIO_BASE_PATH / df.path.map(lambda x: Path(x).name))
    .assign(sentence=lambda df: df.transcript.map(remove_punct))
)

# Test set on specfic Oppday event
ong_test_df = (
    pd.concat([
        pd.read_csv(str(TEST_PATH / "test_label.csv")),
    ])
    .assign(full_path=lambda df: TEST_AUDIO_BASE_PATH / df.filename.map(lambda x: Path(x).name))
    .assign(sentence=lambda df: df["Actual-transcript"].map(remove_punct))    
)

gow_df = (
    pd.DataFrame(gow_data, columns=["filename", "sentence"])
    .assign(full_path=lambda df: GOWAJEE_AUDIO_BASE_PATH / df.filename)
    .assign(sentence=lambda df: df.sentence.map(remove_punct)) 
)

common_voice_train = pd.concat([common_voice_train, common_voice_eval, common_voice_test])[:TRAIN_MAX_N_FILES]
amm_opp_data_df = amm_opp_data_df
eval_set = ong_test_df
gow_df = gow_df

# check size
print(f"common_voice_train size: {len(common_voice_train)}")
print(f"amm_opp_data_df size: {len(amm_opp_data_df)}")
print(f"ong_test_df size: {len(ong_test_df)}")
print(f"gow_df size: {len(gow_df)}")


# In[50]:


def get_file_size(p) -> float:
    return p.stat().st_size / 1024 # kB

def filter_low_size_data(df: pd.DataFrame, filesize_thres=10) -> pd.DataFrame:
    df = df.assign(file_exist=lambda df: df.full_path.map(lambda s: s.exists()))
    df = df[df.file_exist]
    # print(df[~df.file_exist])
    df = df.assign(filesize=lambda df: df.full_path.map(get_file_size))
    df = df[(df.filesize > filesize_thres) & (df.sentence.str.len() > 3)]
    return df


# In[51]:


common_voice_train = filter_low_size_data(common_voice_train, 10)
amm_opp_data_df = filter_low_size_data(amm_opp_data_df, 30)
ong_test_df = filter_low_size_data(ong_test_df, 10)
gow_df = filter_low_size_data(gow_df, 10)

# check size
print(f"common_voice_train size: {len(common_voice_train)}")
print(f"amm_opp_data_df size: {len(amm_opp_data_df)}")
print(f"ong_test_df size: {len(ong_test_df)}")
print(f"gow_df size: {len(gow_df)}")


# In[ ]:


gow_df.head(2)


# In[16]:


print_gpu_info()


# ## Preprocessor prep

# In[23]:


processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3",
    language="Thai",
    task="transcribe",
  )


# In[24]:


def prepare_dataset(batch: list[dict[str, Any]]):
    # load and resample audio data from 48 to 16kHz
    audios = batch["audio"]
    arrays = list(map(lambda a: a["array"], audios))
    sampling_rates = list(map(lambda a: a["sampling_rate"], audios))
    input_lengths = list(map(lambda a: len(a["array"])/a["sampling_rate"], audios))
    sentences = batch["sentence"]

    # compute log-Mel input features from input audio array
    input_features = processor.feature_extractor(
        arrays,
        sampling_rate=AUDIO_SAMPLING_RATE,
    )
    batch["input_features"] = input_features.input_features
    batch["input_lengths"] = input_lengths

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(sentences, padding=True).input_ids
    return batch


# In[25]:


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[Dict[str, list[int] | torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # # if bos token is appended in previous tokenization step,
        # # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
def is_audio_in_length_range(data):
    return data["input_lengths"] < MAX_INPUT_LENGTH

def is_label_in_length_range(data):
    return len(data["labels"]) < MAX_LABEL_LENGTH


# In[26]:


# https://huggingface.co/docs/datasets/audio_dataset#local-files


# In[27]:


def gen_dataset(df: pd.DataFrame) -> Dataset:
    
    return (
        Dataset
        .from_dict({"audio": list(map(str, df.full_path.tolist())), "sentence": df.sentence.tolist(), "path": list(map(str, df.full_path.tolist()))})
        .cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
        .cast_column("sentence", datasets.Value("string"))
    )


# In[28]:


_cmv_set = gen_dataset(common_voice_train)
_oppday_set = gen_dataset(amm_opp_data_df)
_gow_set = gen_dataset(gow_df)

_train_set = interleave_datasets([_cmv_set, _gow_set, _oppday_set], probabilities=TRAIN_SET_DIST, seed=556)
_val_set = gen_dataset(eval_set)


# In[39]:


# tmp = iter(_train_set)
# next(tmp)


# In[35]:


# next(iter(_val_set))


# In[31]:


def load_or_new_process(dataset, config, train_val = "train"):
    
    base_set = dataset.to_iterable_dataset(num_shards=NUM_SHARDS)
    
    if train_val == "train":
        return (
            base_set
            .shuffle(seed=SEED)
            .map(prepare_dataset, batched=True, batch_size=DATA_PROCESS_BATCH_SIZE)
            .filter(is_audio_in_length_range)
            .filter(is_label_in_length_range)
        )
    elif train_val == "val":
        return (
            base_set
            .map(prepare_dataset, batched=True, batch_size=DATA_PROCESS_BATCH_SIZE)
            .filter(is_audio_in_length_range)
            .filter(is_label_in_length_range)        
        )
    
    # Test set case
    return (
        base_set
        .map(prepare_dataset, batched=True, batch_size=DATA_PROCESS_BATCH_SIZE)      
    )


# In[32]:


if not DATASET_CACHE_DIR.exists(): DATASET_CACHE_DIR.mkdir(exist_ok=True)

train_set = load_or_new_process(_train_set, config, "train")
val_set = load_or_new_process(_val_set, config, "val")


# In[33]:


# next(iter(train_set))


# In[28]:


# For debugging
# next(iter(train_set.map(lambda x: x, batched=True)))


# In[29]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[30]:


print_gpu_info()


# # Metrics

# In[31]:


CLEAN_PATTERNS = "((นะ)?(คะ|ครับ)|เอ่อ|อ่า)"
REMOVE_TOKENS = {"", " "}

def hack_wer(
    hypothesis: str,
    reference: str,
    debug=False,
  ) -> float:
    """
    we will tokenize TH long txt into list of words,
    then concat it back separated by whitespace.
    Then, we will just use normal WER jiwer, to utilize
    C++ implementation.
    """
    refs = tokenize(re.sub(CLEAN_PATTERNS, "", reference))
    hyps = tokenize(re.sub(CLEAN_PATTERNS, "", hypothesis))

    refs = [r for r in refs if r not in REMOVE_TOKENS]
    hyps = [h for h in hyps if h not in REMOVE_TOKENS]

    if debug: print(refs); print(hyps)

    return jiwer.wer(" ".join(refs), " ".join(hyps))


def isd_np(preds: list[str], actuals: list[str], debug=True) -> int:
    dp = np.array([np.arange(len(preds) + 1) for _ in range(len(actuals) + 1)], dtype="int16")

    for row in range(len(dp)):
        for col in range(len(dp[0])):
            if row == 0 or col == 0:
                dp[row][col] = max(row, col)
                continue

            if preds[col - 1] != actuals[row - 1]:
                dp[row][col] = min(dp[row - 1][col], dp[row][col - 1], dp[row - 1][col - 1]) + 1
            else:
                dp[row][col] = min(dp[row - 1][col], dp[row][col - 1], dp[row - 1][col - 1])

    if debug: print(*dp, sep="\n")

    return dp[-1][-1]


def wer(pred: str, actual: str, **kwargs) -> float:
    refs = tokenize(re.sub(CLEAN_PATTERNS, "", actual))
    hyps = tokenize(re.sub(CLEAN_PATTERNS, "", pred))

    actuals = [r for r in refs if r not in REMOVE_TOKENS]
    preds = [h for h in hyps if h not in REMOVE_TOKENS]
    if kwargs["debug"]: print(f"{preds}\n{actuals}")
    err = isd_np(preds, actuals, **kwargs)
    return err / len(actuals)


# In[32]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    # pred_str, and label_str is list[str]
    pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wers = list(map(hack_wer, pred_strs, label_strs))
    wer = sum(wers) * 100 / len(wers)

    return {"wer": wer}


# In[33]:


print(hack_wer("สวัสดีครับอิอิ ผมไม่เด็กแล้วนะครับ จริงๆนะ", "สวัสดีครับอุอุ ผมโตแล้วครับ จริงๆนะ", debug=True))
print(wer("สวัสดีครับอิอิ ผมไม่เด็กแล้วนะครับ จริงๆนะ", "สวัสดีครับอุอุ ผมโตแล้วครับ จริงๆนะ", debug=True))


# In[34]:


print_gpu_info()


# # Model prep

# In[35]:


model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH_OR_URL,
    # torch_dtype=torch_dtype,
    num_beams=NUM_BEAMS,
)


# In[36]:


# model config setting
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.enable_input_require_grads()

# Regularization
model.config.dropout = DROPOUT
model.config.apply_spec_augment = APPLY_SPEC_AUGMENT
model.config.mask_feature_length = MASK_FEATURE_LENGTH
model.config.mask_feature_min_masks= MASK_FEATURE_MIN_MASKS
model.config.mask_feature_prob = MASK_FEATURE_PROB
model.config.mask_time_length = MASK_TIME_LENGTH
model.config.mask_time_min_masks = MASK_TIME_MIN_MASKS
model.config.mask_time_prob = MASK_TIME_PROB


# In[37]:


print_gpu_info()


# # Fine-tune the model

# In[38]:


(output_dir := MODEL_CHECKPOINT_DIR).mkdir(exist_ok=True)


# In[39]:


training_args = Seq2SeqTrainingArguments(
    # set to tmp_trainer folder in current folder
    output_dir=str(output_dir),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=N,  # increase by 2x for every 2x decrease in batch size
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS, # 1000
    max_steps=MAX_STEPS, # 6000
    gradient_checkpointing=True,
    fp16=True,
    tf32=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENGTH, 
    save_steps=SAVE_STEPS, # 1000
    eval_steps=EVAL_STEPS, # 1000
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    optim=OPTIMIZER,
)


# In[40]:


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# In[41]:


print_gpu_info()


# In[42]:


result = trainer.train()


# In[ ]:


model.save_pretrained(FINAL_MODEL_OUTPUT_DIR)
processor.save_pretrained(FINAL_PROC_OUTPUT_DIR)


# # Inference

# In[30]:


# # Load from checkpoint or full fine-tune
# processor = WhisperProcessor.from_pretrained(
#     "openai/whisper-large-v3",
#     language="Thai",
#     task="transcribe",
#   )

# model = WhisperForConditionalGeneration.from_pretrained(
#     MODEL_PATH_OR_URL,
#     # torch_dtype=torch_dtype,
#     num_beams=NUM_BEAMS,
# )


# In[31]:


model.config.dropout=0.0
pipe = pipeline(
    "automatic-speech-recognition",
    model=model, tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
  )


# In[ ]:


preds = pipe([str(p) for p in common_voice_test.full_path.tolist()], generate_kwargs={"language":"<|th|>", "task":"transcribe"}, batch_size=BATCH_SIZE)


# In[ ]:


outputs = []

for pred, t in zip(preds, common_voice_test.itertuples()):
    outputs.append((pred["text"], t.sentence, t.full_path))


# In[ ]:


df = pd.DataFrame(outputs, columns=["pred", "actual", "fp"])
print(df.head())


# In[ ]:


df.to_excel("test-results.xlsx")


# # Eval
