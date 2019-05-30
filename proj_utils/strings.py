# -*- coding: utf-8
import re

EN_PATTERN = re.compile('[A-Za-z<>]+')
NUM_PATTERN = re.compile('[0-9]')

def q2b(q_string):
    """全角转半角"""
    b_list = []
    for ch in q_string:
        code = ord(ch)
        if code == 12288: # 空格
            code = 32
        elif 65281 <= code <= 65374:
            code -= 65248
        elif code == 58380:  # \ue40c \n -> 空格
            code = 32
        b_list.append(chr(code))
    return ''.join(b_list)


def is_enword(string):
    global EN_PATTERN
    if EN_PATTERN.fullmatch(string):
        return True
    return False


def has_number(string):
    global NUM_PATTERN
    return NUM_PATTERN.search(string)