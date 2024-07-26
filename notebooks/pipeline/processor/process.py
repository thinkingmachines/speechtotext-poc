# flake8: noqa: E722
import re
from typing import List

import deepcut
from langdetect import detect

from .english import EnglishTextNormalizer
from .postfix import remove_longest_repeating_words


def en_preprocess(text: str) -> str:
    text = re.sub("[QA]:|(uh)", "", text)
    return text


def th_preprocess(text: str) -> str:
    text = re.sub("((ค่ะ)|((นะ)?ครับ)|(นะคะ)|เอ่อ)+", "", text)
    # text = thaispellcheck.check(text, autocorrect=True)
    return text


def pred_postprocess(listtext: List[str]) -> List[str]:
    if " " in listtext:
        listtext.remove(" ")

    clean_text = remove_longest_repeating_words(listtext)  # changed
    return clean_text


def tokenize_text(text: str, pred=False) -> list:
    if not isinstance(text, str):
        return []

    try:
        lang = detect(text)
    except:
        lang = "en"

    if lang != "en":
        text = th_preprocess(text)
        splited_text = deepcut.tokenize(text)
    else:
        eng_normalizer = EnglishTextNormalizer()
        text = eng_normalizer(text)
        text = en_preprocess(text)
        splited_text = text.split(" ")

    return pred_postprocess(splited_text) if pred else splited_text
