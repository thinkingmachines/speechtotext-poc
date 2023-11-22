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
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
from itertools import compress

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import datasets
import torch
import jiwer
import numpy as np
import pandas as pd
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig
# from deepcut import tokenize  # Consume too much memory when using with CUDA
from pythainlp.tokenize import word_tokenize as tokenize
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, pipeline, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration
from datasets import Dataset, load_from_disk
from datasets.features import Audio


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
TRAIN_SIZE = 0.99
SEED = 4242
SAMPLING = 1  # sampling rate
AUDIO_SAMPLING_RATE = 16_000
MODEL_PATH_OR_URL = "openai/whisper-large-v3"

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
LEARNING_RATE = 1e-6
WARMUP_STEPS = 1000
MAX_STEPS = 5000
SAVE_STEPS = 1000
EVAL_STEPS = 500
LOGGING_STEPS = 25
EVAL_MAX_N_FILES = 500
TRAIN_MAX_N_FILES = None
OPTIMIZER = "adamw_bnb_8bit"
DROPOUT = 0.1

MAX_LABEL_LENGTH = 448
GENERATION_MAX_LENGTH = 225
CHUNK_LENGTH = 30
NUM_BEAMS = 1
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
N = 4

# resource config
MODEL_CHECKPOINT_DIR = Path.cwd() / "fine-tune-whisper-large-v3-checkpoints"
DATASET_CACHE_DIR = Path.cwd() / "dataset-cache"
FINAL_MODEL_OUTPUT_DIR = Path.cwd() / "fine-tune-model"

PRJ_ROOT = Path.cwd().parents[2]
DATA_PATH = PRJ_ROOT / "notebooks" / "whisper-v3" / "data"
COMMON_VOICE_PATH = DATA_PATH /  "cv-corpus-15" / "th"
AUDIO_BASE_PATH = COMMON_VOICE_PATH / "clips"


# In[4]:


def remove_punct(s: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", s)

remove_punct("ไหน?ลองซิ! ...")


# In[5]:


common_voice_metadata = (
    pd.concat([
        pd.read_csv(str(COMMON_VOICE_PATH / "other.tsv"), sep="\t"),
        pd.read_csv(str(COMMON_VOICE_PATH / "validated.tsv"), sep="\t"),
        pd.read_csv(str(COMMON_VOICE_PATH / "invalidated.tsv"), sep="\t"),
    ])
    .assign(full_path=lambda df: AUDIO_BASE_PATH / df.path)
    .assign(sentence=lambda df: df.sentence.map(remove_punct))
)


# In[6]:


shuffled_common_voice = common_voice_metadata.sample(frac=SAMPLING, random_state=SEED) # shuffling
common_voice_train, common_voice_eval = train_test_split(shuffled_common_voice, test_size=(1 - TRAIN_SIZE))


# In[7]:


print_gpu_info()


# ## Preprocessor prep

# In[8]:


processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3",
    language="Thai",
    task="transcribe",
  )


# In[9]:


def prepare_dataset(batch: list[dict[str, Any]]):
    # load and resample audio data from 48 to 16kHz
    audios = batch["audio"]
    arrays = list(map(lambda a: a["array"], audios))
    sampling_rates = list(map(lambda a: a["sampling_rate"], audios))
    sentences = batch["sentence"]

    # compute log-Mel input features from input audio array
    input_features = processor.feature_extractor(
        arrays,
        sampling_rate=AUDIO_SAMPLING_RATE,
    )
    batch["input_features"] = input_features.input_features

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(
        sentences, padding="longest", truncation=True, max_length=MAX_LABEL_LENGTH
    ).input_ids
    return batch


# In[10]:


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
        # if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
        #     labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[11]:


# https://huggingface.co/docs/datasets/audio_dataset#local-files


# In[12]:


def gen_dataset(df: pd.DataFrame) -> Dataset:
    return (
        Dataset
        .from_dict({"audio": list(map(str, df.full_path.tolist())), "sentence": df.sentence.tolist(), "path": list(map(str, df.full_path.tolist()))})
        .cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
        .cast_column("sentence", datasets.Value("string"))
    )


# In[13]:


train_set = gen_dataset(common_voice_train.iloc[:TRAIN_MAX_N_FILES])
val_set = gen_dataset(common_voice_eval.iloc[:EVAL_MAX_N_FILES])


# In[14]:


def load_or_new_process(dataset, config, train_val = "train"):
    
#     name = f"{MAX_STEPS * BATCH_SIZE // 1_000}k-size-{train_val}"
    
#     if (DATASET_CACHE_DIR / train_val).exists():
#         dataset = load_from_disk(DATASET_CACHE_DIR / name)
#     else:
#         dataset = dataset.map(
#             prepare_dataset, 
#             cache_file_name=str(DATASET_CACHE_DIR / f"{name}.pt"),
#             **config,
#         )
#         dataset.save_to_disk(DATASET_CACHE_DIR / name)
    
    dataset = dataset.with_transform(prepare_dataset)
    return dataset


# In[15]:


if not DATASET_CACHE_DIR.exists(): DATASET_CACHE_DIR.mkdir(exist_ok=True)

train_set = load_or_new_process(train_set, config, "train")
val_set = load_or_new_process(val_set, config, "val")


# In[16]:


# For debugging
# next(iter(train_set.map(lambda x: x, batched=True)))


# In[17]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[18]:


print_gpu_info()


# # Metrics

# In[19]:


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


# In[20]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    # pred_str, and label_str is list[str]
    pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    with mp.Pool(8) as pool:
        
        wers = list(pool.starmap(hack_wer, zip(pred_strs, label_strs)))
        wer = sum(wers) * 100 / len(wers)

    return {"wer": wer}


# In[21]:


print(hack_wer("สวัสดีครับอิอิ ผมไม่เด็กแล้วนะครับ จริงๆนะ", "สวัสดีครับอุอุ ผมโตแล้วครับ จริงๆนะ", debug=True))
print(wer("สวัสดีครับอิอิ ผมไม่เด็กแล้วนะครับ จริงๆนะ", "สวัสดีครับอุอุ ผมโตแล้วครับ จริงๆนะ", debug=True))


# In[22]:


print_gpu_info()


# # Model prep

# In[23]:


model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH_OR_URL,
    # torch_dtype=torch_dtype,
    num_beams=NUM_BEAMS,
)


# In[24]:


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.drop_out = DROPOUT
model.enable_input_require_grads()


# In[25]:


print_gpu_info()


# # Fine-tune the model

# In[26]:


(output_dir := MODEL_CHECKPOINT_DIR).mkdir(exist_ok=True)


# In[27]:


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
    eval_steps=SAVE_STEPS, # 1000
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    optim=OPTIMIZER,
)


# In[28]:


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# In[ ]:


result = trainer.train()


# In[ ]:


metric = result.metrics


print(result, end="\n\n")
print(f"GPU Flops perf: {metric['total_flos'] / metric['train_runtime'] / (1024**4)} TFLOPS")


# In[ ]:


model.save_pretrained(FINAL_MODEL_OUTPUT_DIR)


# # Inference

# In[ ]:


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


# In[ ]:


pipe = pipeline(
    "automatic-speech-recognition",
    model=model, tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
  )


# In[ ]:


pipe(val_set[-2]["path"])


# In[ ]:


val_set[-2]["sentence"]


# # Eval
