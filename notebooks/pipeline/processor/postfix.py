from typing import List


def keys_with_max_value(d):
    max_value = max(d.values())
    max_keys = [k for k, v in d.items() if v == max_value]
    return max_keys


def remove_duplicate_keep_first(
    word_list: List[str], duplicate_word: List[str]
) -> List[str]:
    unique_list = []
    duplicate_word = set(duplicate_word)
    for word in word_list:
        if word not in duplicate_word:
            unique_list.append(word)
        elif word in unique_list:
            continue
        else:
            unique_list.append(word)
    return unique_list


def remove_longest_repeating_words(word_list: List[str]) -> List[str]:
    if len(set(word_list)) == len(word_list):
        return word_list

    word_count = {}
    for word in word_list:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    print(word_count)
    repeat_words = [wd for wd, cnt in word_count.items() if cnt > 3]
    if repeat_words:
        print(f"repeating words = {repeat_words}")
        return remove_duplicate_keep_first(word_list, repeat_words)
    else:
        return word_list
