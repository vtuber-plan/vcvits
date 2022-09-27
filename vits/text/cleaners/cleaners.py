import re
from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')
_dot_re = re.compile(r'\.+')

def lowercase(text: str) -> str:
  return text.lower()

def collapse_whitespace(text: str) -> str:
  return re.sub(_whitespace_re, ' ', text)

def collapse_dot(text: str) -> str:
  return re.sub(_dot_re, ' ', text)

def convert_to_ascii(text: str) -> str:
  return unidecode(text)
