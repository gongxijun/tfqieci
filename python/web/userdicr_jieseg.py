#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     userdicr_jieseg.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/22
-----------------------------------
Change Activiy  :   2018/11/22
-----------------------------------

"""
__author__ = 'xijun1'
import jieba

#jieba.load_userdict("/data0/xijun1/tfqieci/python/basic_vocabutf8.txt")
import jieba.posseg as pseg


class UserDictSeg:
    def __init__(self):
        pass

    def sentence(self, text):
        return u"||".join(jieba.cut(text))
