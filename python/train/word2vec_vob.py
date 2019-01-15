#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     word2vec_vob.py
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


class Word2vecVocab:
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
            secondToken = ""
	    for t, line in enumerate(fp):
                line = line.strip()
                ss = line.split(" ")
                assert (len(ss) == (dim + 1))
		self.word2id[ ss[0] ] = t
		if len( self.word2id ) ==1:
		    assert( ss[0] == "</s>"),"first tok should be </s>"
		elif len( self.word2id ) ==2:
		    secondToken = ss[0]
	        if ss[0] in ["<unk>","<UNK>"]:
		    self.word2id[ ss[0] ] = 1
		    self.word2id[ secondToken ] = t
		    secondToken = ss[0]
			
                if 0 == t % 50000:
                    print("loading  cur_word: {} ,total_word: {} ,percent: {}"
                          "".format(t, total, float(t) / float(total)))
	    assert secondToken in ["<unk>","<UNK>"] , "second token should be '<UNK>' "
	    		
    def GetWordIndex(self, word):
        """
         返回index
        :param word:
        :return:
        """
        if word in self.word2id:
            return self.word2id[word]
        return self.word2id["<UNK>"]
   
    def DumpBasicVocab(self, out_path):
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
            #fw.write("{}\t{}\n".format("<UNK>", total))

