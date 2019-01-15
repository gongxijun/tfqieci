#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     sentence_breaker.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/13
-----------------------------------
Change Activiy  :   2018/11/13
-----------------------------------

"""
__author__ = 'xijun1'


class SentenceBreaker:
    def __init__(self, maxLen):
        self.kInlineMarks = [u"（", u"）", u"(", u")", u"[", u"]", u"【", u"】u", u"《", u"》", u"“", u"”"]
        self.kBreakMarks = [u"。", u",", u"，", u" ",u"\t", u"?", u"？", u"!", u"！", u";", u"；"]
        self.max_len_ = maxLen
        self.inline_marks_set_ = set()
        self.break_marks_ = set()
        # put inline
        for i in range(1, len(self.kInlineMarks), 2):
            self.inline_marks_set_.add((self.kInlineMarks[i - 1], self.kInlineMarks[i]))
        # put break marks
        for i in range(len(self.kBreakMarks)):
            self.break_marks_.add(self.kBreakMarks[i])

    def is_inline_mark(self, param):
        if param not in self.inline_marks_set_:
            return False
        return True

    def is_break_mark(self, param):
        if param not in self.break_marks_:
            return False
        return True

    def breakSentences(self, text, lines):
        markChar = 0
        text=text.decode('utf8')
        text_size = len(text)
        if text_size == 0:
            print " text size  is ", text_size
            return lines
        markPos = 0
        for i in range(text_size):
            word = text[i]
            if self.is_inline_mark(word):
                if markChar == word:
                    lines.append(text[markPos: i  + 1])
                    markPos = i + 1
                    markChar = 0
                else:
                    if markPos != i:
                        lines.append(text[markPos: i ])
                        markPos = i
                    markChar = self.inline_marks_set_[word]
            elif markChar == 0:
                if self.is_break_mark(word) or (i - markPos + 1) >= self.max_len_:
                    lines.append(text[markPos: i + 1])
                    markPos = i + 1
            elif (i - markPos + 1) >= self.max_len_:
                lines.append(text[markPos: i + 1])
                markPos = i + 1
                markChar = 0
        if markPos < text_size:
            lines.append(text[markPos:  text_size])
        return lines
