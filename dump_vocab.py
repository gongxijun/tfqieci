# -*- coding: utf-8 -*-
# @Author: Koth
# @Date:   2016-11-20 15:04:18
# @Last Modified by:   Koth
# @Last Modified time: 2016-11-20 15:07:51
import sys
import os
sys.path.insert(0,"/data0/xijun1/tfqieci/python/train")
import  word2vec_vob as w2v
def main(argc, argv):
  if argc < 3:
    print("Usage:%s <word2vec_vocab_path> <output_path>" % (argv[0]))
    sys.exit(1)
  vob = w2v.Word2vecVocab()
  vob.load(argv[1])
  vob.DumpBasicVocab(argv[2])


if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
