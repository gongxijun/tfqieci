#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     basic_vocab.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/30
-----------------------------------
Change Activiy  :   2018/11/30
-----------------------------------

"""
__author__ = 'xijun1'
import codecs
import numpy as np


class CaseInsensitiveDict(dict):
    """
     忽略大小写的字符串
    """

    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(key.lower())


class BasicVocab:
    def __init__(self):
        self.word2id = CaseInsensitiveDict()
        pass

    def load(self, mode_path):
        """
        load word2vec model
        :param mode_path: model path
        :return:
        """
        with  codecs.open(mode_path, "r") as fp:
            print("load data from:", mode_path)
            line = next(fp)
            line = line.strip()
            ss = line.split(" ")
            total = int(ss[0])
            dim = int(ss[1])
            print("dim : {} words: {} ".format(dim, total))
	    total_char =0
            for t, line in enumerate(fp):
                line = line.strip()
                ss = line.split(" ")
                assert (len(ss) == (dim + 1))
                for _char in ss[0]:
		   if _char not in self.word2id:
			self.word2id[ _char  ] = total_char
			total_char += 1
                if 0 == t % 50000:
                    print("loading  cur_word: {} ,total_word: {} ,percent: {}"
                          "".format(t, total, float(t) / float(total)))

    def GetWordIndex(self, word):
        """
         返回index
        :param word:
        :return:
        """
        if word not in self.word2id:
            self.word2id[ word ] = len( self.word2id) 
        
	return self.word2id[ word ]
   
    def DumpBasicVocab(self, out_path="./basic_vocab.txt"):
        """

        :param mode_path:  word2vec model
        :param out_path:  word_id output path
        :return:
        """
	total = len( self.word2id)
        with  codecs.open(out_path,mode="w") as fw:
            for t , (_char , ind) in enumerate( self.word2id.items()):       
		   if 0 == t % 50000:
                        print("loading  cur_word: {} ,total_word: {} ,percent: {}"
                              "".format(t, total, float(t) / float(total)))
                    fw.write("{}\t{}\n".format(_char, ind))
            fw.write("{}\t{}\n".format("<UNK>", total))

