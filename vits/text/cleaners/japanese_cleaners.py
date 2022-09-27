

from typing import List
from .japanese_mapping import ROMAJI_LIST
from .cleaners import convert_to_ascii, lowercase, collapse_dot, collapse_whitespace

def split_romaji(text: str) -> List[str]:
    out = []
    left_text = text
    while len(left_text) > 0:
        not_found = True
        for c in ROMAJI_LIST:
            if left_text.startswith(c):
                out.append(c)
                left_text = left_text[len(c):]
                not_found = False
                break
        if not_found:
            out.append(left_text[-1])
            left_text = left_text[1:]
    return out

def japanese_cleaners(text):
    '''Pipeline for Japanese text, including abbreviation expansion. + punctuation + stress'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_dot(text)
    phonemes = collapse_whitespace(text)
    return phonemes