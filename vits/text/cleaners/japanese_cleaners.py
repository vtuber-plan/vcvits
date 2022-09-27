
def japanese_cleaners(text):
    '''Pipeline for Japanese text, including abbreviation expansion. + punctuation + stress'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_dot(text)
    
    out = ""
    for c in text:
        if c in symbols:
            out += c
        else:
            pass
    phonemes = collapse_whitespace(out)
    return phonemes