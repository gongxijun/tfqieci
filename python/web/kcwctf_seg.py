#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     jieba_seg.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/22
-----------------------------------
Change Activiy  :   2018/11/22
-----------------------------------

"""
__author__ = 'xijun1'
import sys,os
sys.path.append("/data0/xijun1/tfqieci/python")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_seg_model import TfSegModel

# dstr = "安装好tensorflow,切换到kcws代码目录";
# print dstr.decode('utf8')[0:3].encode('utf8')
tfmodel_ = TfSegModel();
basic_vocab = '/data0/xijun1/tfqieci/models/bbasic_vocab.txt'
model_dir = '/data0/xijun1/tfqieci/logs/'
tfmodel_.LoadModel(model_dir, basic_vocab, 200, None)
pTopResults = list()


class KCWCSeg:
    def __init__(self):
        pass

    def sentence(self, text):
        pTopResults=[]
        pTopResults = tfmodel_.SegmentpTags(text.encode('utf8'), pTopResults, None)
        return u"||".join(pTopResults)
