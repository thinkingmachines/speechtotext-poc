#!/usr/bin/env python
# coding: utf-8

# In[138]:


# # Download ong's data
# !gdown --id 1ZZJ23ejcMEsaLBkSbOZmlxXyhIVZqfJo

# # Download amm's data
# !gdown --id 1TRPRcDEyi6ogu8j4OE_d_ePJRxUw_X1m


# In[139]:


from pathlib import Path
from itertools import chain
import re

import pysrt
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_extract_audio


# In[140]:


ROOT_PATH = Path.cwd().parent / "data" / "Opp Day SRTs" 
(VIDEO_CHUNKS_PATH := ROOT_PATH / "chunks").mkdir(exist_ok=True)


# In[141]:


fr_srts = list((ROOT_PATH / "Freelancer SRT").rglob("*.srt"))
capcut_srts = list((ROOT_PATH / "Capcut SRT").rglob("*.srt"))
vids = list((ROOT_PATH).rglob("*.mp4"))


# In[142]:


def gen_srt_map(ls: list[Path], exist_dict: dict = {}) -> dict[str, Path]:
    def hard_mapping(s: str) -> str:
        s = Path(clean_logic(s)).stem.upper()
        mapping = {"interpharma": "IP"}
        return mapping.get(s.lower(), s)
        
        
    def clean_logic(s: str) -> str:
        s = re.sub("FL-", "", s)
        if " " in s:
            candidate = s.split(" ")[0]
            if candidate == "Opp":
                return re.search("[A-Z]{2,}", s).group(0)
            return candidate
        
        return s.split("-")[0]
        
    return exist_dict | {hard_mapping(clean_logic(l.name)): l for l in ls}

def gen_video_map(ls: list[Path]) -> dict[str, Path]:
    return {re.search("[A-Z]{2,}", l.name).group(0): l for l in ls}

def cut_video_into_chunks(symbol: str, srt_maps: dict, vid_maps: dict) -> list[str, Path]:
    sub_pth, vid_pth = srt_maps.get(symbol, None), vid_maps.get(symbol, None)
    
    if sub_pth is None or vid_pth is None:
        print("sub or vid is not exist", sub_pth, vid_pth)
        return

    subs = pysrt.open(sub_pth)
    
    metadata = []
    # Loop through the subtitles
    for i, sub in enumerate(subs):
        print(sub)
        # Get the start and end times of each subtitle
        start_time = sub.start.ordinal / 1000.0
        end_time = sub.end.ordinal / 1000.0

        # Use ffmpeg to cut the video for this subtitle
        target_pth = VIDEO_CHUNKS_PATH / f"{vid_pth.name}_{i:04d}.mp4"
        target_audio_pth = VIDEO_CHUNKS_PATH / f"{vid_pth.name}_{i:04d}.mp3"
        transcription = sub.text
        ffmpeg_extract_subclip(vid_pth, start_time, end_time, targetname=target_pth)
        ffmpeg_extract_audio(target_pth, target_audio_pth)
    
        metadata.append((symbol, target_audio_pth, transcription))
        
    return metadata
        


# In[143]:


srt_maps = gen_srt_map(capcut_srts)
srt_maps = gen_srt_map(fr_srts, srt_maps)


# In[144]:


vid_maps = gen_video_map(vids)


# In[145]:


print(srt_maps.keys())
print(vid_maps.keys())


# In[146]:


target_symbols = set(vid_maps.keys()).intersection(srt_maps.keys())
print(target_symbols)


# In[147]:


results = list(chain.from_iterable([cut_video_into_chunks(symbol, srt_maps, vid_maps) for symbol in target_symbols]))


# In[148]:


df = pd.DataFrame(results, columns=["symbol", "path", "transcript"])


# In[149]:


df.to_csv(ROOT_PATH / "train_metadata.tsv", sep="\t", index=False)

