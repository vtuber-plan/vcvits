from .cleaners import collapse_whitespace

from pypinyin import pinyin, lazy_pinyin, Style

def replace_chinese_mark(text: str) -> str:
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("？", "?")
    text = text.replace("！", "!")
    text = text.replace("、", ",")
    text = text.replace("「", "\"")
    text = text.replace("」", "\"")
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    return text

def chinese_cleaners(text: str):
    '''Pipeline for Chinese text'''
    text = replace_chinese_mark(text)
    ret = pinyin(text, style=Style.TONE3, heteronym=True)
    out = "_".join([c[0] for c in ret])
    phonemes = collapse_whitespace(out)
    return phonemes
