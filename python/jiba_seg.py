#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     jiba_seg.py
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
import codecs

# with codecs.open("/Users/sina/github/python_demo/basic_vocabutf8.txt","wb") as fw:
#     with open("/Users/sina/github/python_demo/basic_vocab.txt","r") as fr:
#         for line in fr :
#             fw.write( line.split("\t")[0]+"\n")


jieba.load_userdict("/Users/sina/github/python_demo/basic_vocabutf8.txt")
import jieba.posseg as pseg
from flask import Flask

app = Flask(__name__)

test_sent = (
    "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
    "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)


@app.route("/")
def hello():
    words = jieba.cut(test_sent)
    print('/'.join(words))
    return "Hello World!"
