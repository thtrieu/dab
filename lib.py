from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import bleu_hook


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import collections
import math
import os
import re
import sys
import time
import unicodedata
import numpy as np
import pandas as pd
import six
from datetime import datetime
import tqdm


def check_mrs(content, i):
  is_mr = (i >= 2 and 
           content[i-2:i].lower() in ['mr', 'ms'] and
           (i < 3 or content[i-3] == ' '))
  is_mrs = (i >= 3 and 
            content[i-3:i].lower() == 'mrs' and 
            (i < 4 or content[i-4] == ' '))
  return is_mr or is_mrs


def check_ABB_mid(content, i):
  if i <= 0:
    return False
  if i >= len(content)-1:
    return False
  l, r = content[i-1], content[i+1]
  return l.isupper() and r.isupper()


def check_ABB_end(content, i):
  if i <= 0:
    return False
  l = content[i-1]
  return l.isupper()


def fix_file(filename):
  if not os.path.exists(filename + '.fixed'):
    with open(filename, 'r') as file:
      contents = file.read()
    contents = fix_contents(contents)

    with open(filename+'.fixed', 'w') as file:
      file.write(contents)

  return filename + '.fixed'


def fix_contents(contents):
  # first step: replace special characters 
  check_list = ['\uFE16', '\uFE15', '\u0027','\u2018', '\u2019',
                '“', '”', '\u3164', '\u1160', 
                '\u0022', '\u201c', '\u201d', '"',
                '[', '\ufe47', '(', '\u208d',
                ']', '\ufe48', ')' , '\u208e', 
                '—', '_', '–', '&']
  alter_chars = ['?', '!', '&apos;', '&apos;', '&apos;',
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&#91;', '&#91;', '&#91;', '&#91;',
                 '&#93;', '&#93;', '&#93;', '&#93;', 
                 '-', '-', '-', '&amp;']

  replace_dict = dict(zip(check_list, alter_chars))

  new_contents = ''
  for i, char in enumerate(contents):
    #    total=len(contents):
    if char == '&' and (contents[i:i+5] == '&amp;' or
                        contents[i:i+6] == '&quot;' or
                        contents[i:i+6] == '&apos;' or
                        contents[i:i+5] == '&#93;' or
                        contents[i:i+5] == '&#91;'):
      new_contents += char
      continue
    new_contents += replace_dict.get(char, char)
  contents = new_contents

  # second: add spaces
  check_sp_list = [',', '?', '!', '&apos;', '&amp;', '&quot;', '&#91;', 
                   '&#93;', '-', '/', '%', ':', '$', '#', '&', '*', ';', '=', '+', '$', '#', '@', '~', '>', '<']

#   print('[2/4]')
  new_contents = ''
  i = 0
  #   l100 = len(contents)//100
  while i < len(contents):
    # if i // l100 > (i-1) // l100:
    #   sys.stdout.write(str(i // l100) + '%')
    #   sys.stdout.flush()
    char = contents[i]
    found = False
    for string in check_sp_list:
      if string == contents[i: i+len(string)]:
        new_contents += ' ' + string 
        if string != '&apos;':
          new_contents += ' '
        i += len(string)
        found = True
        break
    if not found:
      new_contents += char
      i += 1
  contents = new_contents

#   print('[3/4]')
  new_contents = ''
  for i, char in enumerate(contents):
    #   , total=len(contents)):
    if char != '.':
      new_contents += char
      continue
    elif check_mrs(contents, i):
      # case 1: Mr. Mrs. Ms.
      new_contents += '. '
    elif check_ABB_mid(contents, i):
      # case 2: U[.]S.A.
      new_contents += '.'
    elif check_ABB_end(contents, i):
      # case 3: U.S.A[.]
      new_contents += '. '
    else:
      new_contents += ' . '

  contents = new_contents
  
  # third: remove not necessary spaces.
#   print('[4/4]')
  new_contents = ''
  for char in contents:
    if new_contents and new_contents[-1] == ' ' and char == ' ':
      continue
    new_contents += char
  contents = new_contents
  
  return contents.strip()


