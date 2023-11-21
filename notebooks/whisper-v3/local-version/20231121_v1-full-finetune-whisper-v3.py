#!/usr/bin/env python
# coding: utf-8

# # Data prep

# In[15]:


import re
import requests
import string
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

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
from transformers import WhisperProcessor, pipeline, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, load_from_disk
from datasets.features import Audio


# In[16]:


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

# In[17]:


# Data config
TRAIN_SIZE = 0.9
SEED = 4242
SAMPLING = 1  # sampling rate
AUDIO_SAMPLING_RATE = 16_000

# model config
LEARNING_RATE = 0.5e-5
WARMUP_STEPS = 1500
MAX_STEPS = 5000
SAVE_STEPS = 1000
EVAL_STEPS = SAVE_STEPS
LOGGING_STEPS = 200

CHUNK_LENGTH = 30
NUM_BEAMS = 2
BATCH_SIZE = 16
N = 1

# resource config
MODEL_CHECKPOINT_DIR = Path.cwd() / "fine-tune-whisper-large-v3-checkpoints"
DATASET_CACHE_DIR = Path.cwd() / "dataset-cache"
FINAL_MODEL_OUTPUT_DIR = Path.cwd() / "fine-tune-model"

PRJ_ROOT = Path.cwd().parents[2]
DATA_PATH = PRJ_ROOT / "notebooks" / "whisper-v3" / "data"
COMMON_VOICE_PATH = DATA_PATH /  "cv-corpus-15" / "th"
AUDIO_BASE_PATH = COMMON_VOICE_PATH / "clips"


# In[18]:


def remove_punct(s: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", s)

remove_punct("ไหน?ลองซิ! ...")


# In[19]:


common_voice_metadata = (
    pd.concat([
        pd.read_csv(str(COMMON_VOICE_PATH / "other.tsv"), sep="\t"),
        pd.read_csv(str(COMMON_VOICE_PATH / "validated.tsv"), sep="\t"),
        pd.read_csv(str(COMMON_VOICE_PATH / "invalidated.tsv"), sep="\t"),
    ])
    .assign(full_path=lambda df: AUDIO_BASE_PATH / df.path)
    .assign(sentence=lambda df: df.sentence.map(remove_punct))
)


# In[20]:


shuffled_common_voice = common_voice_metadata.sample(frac=SAMPLING, random_state=SEED) # shuffling
common_voice_train, common_voice_eval = train_test_split(shuffled_common_voice, test_size=(1 - TRAIN_SIZE))


# In[21]:


print_gpu_info()


# ## Preprocessor prep

# In[22]:


processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3",
    language="Thai",
    task="transcribe",
  )


# In[23]:


def prepare_dataset(batch: list[dict[str, Any]]):
    # load and resample audio data from 48 to 16kHz
    # print(batch)
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
    batch["labels"] = processor.tokenizer(sentences).input_ids
    return batch


# In[24]:


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

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[25]:


# https://huggingface.co/docs/datasets/audio_dataset#local-files


# In[26]:


def gen_dataset(df: pd.DataFrame) -> Dataset:
    return (
        Dataset
        .from_dict({"audio": list(map(str, df.full_path.tolist())), "sentence": df.sentence.tolist()})
        .cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
        .cast_column("sentence", datasets.Value("string"))
    )


# In[27]:


config = {
    # "num_proc": 2, 
    # "keep_in_memory": False, 
    # "load_from_cache_file": True, 
    # "writer_batch_size": 20,
}

train_set = gen_dataset(common_voice_train)
val_set = gen_dataset(common_voice_eval)

train_set = train_set.with_transform(
    prepare_dataset, 
    **config,
)
val_set = val_set.with_transform(
    prepare_dataset, 
    **config,
)

#     train_set.save_to_disk(DATASET_CACHE_DIR / "train")
#     val_set.save_to_disk(DATASET_CACHE_DIR / "eval")


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
    wer = sum(map(lambda w: w * 100, wers)) / len(wers)

    return {"wer": wer}


# In[33]:


print(hack_wer("สวัสดีครับอิอิ ผมไม่เด็กแล้วนะครับ จริงๆนะ", "สวัสดีครับอุอุ ผมโตแล้วครับ จริงๆนะ", debug=True))
print(wer("สวัสดีครับอิอิ ผมไม่เด็กแล้วนะครับ จริงๆนะ", "สวัสดีครับอุอุ ผมโตแล้วครับ จริงๆนะ", debug=True))


# In[34]:


print_gpu_info()


# # Model prep

# In[35]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# In[37]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    # torch_dtype=torch_dtype,
    num_beams=NUM_BEAMS,
)


# In[38]:


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.enable_input_require_grads()


# In[39]:


print_gpu_info()


# # Fine-tune the model

# In[40]:


(output_dir := MODEL_CHECKPOINT_DIR).mkdir(exist_ok=True)


# In[41]:


training_args = Seq2SeqTrainingArguments(
    # set to tmp_trainer folder in current folder
    output_dir=str(output_dir),
    per_device_train_batch_size=BATCH_SIZE // N,
    gradient_accumulation_steps=N,  # increase by 2x for every 2x decrease in batch size
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS, # 1000
    max_steps=MAX_STEPS, # 6000
    gradient_checkpointing=True,
    fp16=True,
    tf32=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=max(BATCH_SIZE // N // 2, 1),
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_STEPS, # 1000
    eval_steps=SAVE_STEPS, # 1000
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
)


# In[42]:


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


pipe = pipeline(
    "automatic-speech-recognition",
    model=model, tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
  )


# In[ ]:


# Load base model and plugin Peft fine-tuned params to getbak our model
# model = WhisperForConditionalGeneration.from_pretrained(
#     "openai/whisper-large-v3",
#     torch_dtype=torch_dtype,
#     num_beams=NUM_BEAMS,
# )

## Load from finalized model
# model = PeftModel.from_pretrained(model, FINAL_MODEL_OUTPUT_DIR)
## Load from checkpoint
# model = PeftModel.from_pretrained(model, MODEL_CHECKPOINT_DIR / "checkpoint-0008")


# In[ ]:





# # Eval
