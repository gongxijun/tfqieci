#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     tf_seg_model.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/13
-----------------------------------
Change Activiy  :   2018/11/13
-----------------------------------

"""
__author__ = 'xijun1'
import tensorflow as tf
from tfmodel import TfModel
from sentence_breaker import SentenceBreaker
import math
import codecs
import numpy as np
from viterbi_decode import *
import esmre
import esm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("transition_node_name", "transitions", "the transitions node in graph model")
tf.app.flags.DEFINE_string("scores_node_name", "Reshape_7", "the final emission  node in graph model")
tf.app.flags.DEFINE_string("input_node_name", "input_placeholder", "the input placeholder  node in graph model")
tf.app.flags.DEFINE_integer("num_shards" ,4 ,"number of vecs ")
tf.app.flags.DEFINE_integer("embedding_size",200, "embedding size")
tf.app.flags.DEFINE_string("w2v_path","/data0/xijun1/tfqieci/vec.txt", "word2vector path")
basic_vocab = '/data0/xijun1/tfqieci/models/bbasic_vocab.txt'
model_dir = '/data0/xijun1/tfqieci/logs/'

class CaseInsensitiveDict(dict):
    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(key.lower())

def load_vocab(path, pVocab_map):
    """
    load vocab
    :param path:
    :param pVocab_map:
    :return:
    """
    for line in codecs.open(path, mode="rb", encoding="utf-8"):
        line = line.strip("\n")
        line = line.strip("\r")
        arr_words = line.split("\t")
        if len(arr_words) != 2:
            print "line len not comformed to dimension:%s:%d\n".format(line, len(arr_words))
            return
        word = arr_words[0]
        if word in ["</s>"]:
            continue
        if word not in pVocab_map:
            pVocab_map[word] = int(arr_words[1])
    return pVocab_map


class TfSegModel:
    def __init__(self):
        """
        @todo 暂时不增加自定义字典功能
        """
        self.max_sentence_len_ = None
        self.breaker_ = None
        self.model = None
        self.transitions_ = list()
        self.vocab_ = CaseInsensitiveDict()
        self.scores_ = None
        self.bp_ = None
        self.scanner_ = esm.Index()
        self.num_tags_ = 0
        self.tagger_ = None

    def loadUserDict(self, userDictPath):
        """
        自定义字典
        :param userDictPath:
        :return:
        """

        for line in codecs.open(userDictPath, mode="rb", encoding="utf-8"):
            line = line.strip("\n")
            line = line.strip("\r")
            arr_words = line.split("\t")
            if len(arr_words) != 2:
                print "line len not comformed to dimension:%s:%d\n".format(line, len(arr_words))
                return
            word = arr_words[0]
            if word in ["</s>"]:
                continue
    def load_w2v(self,num_shards, path, expectDim):
        with  open(path, "r") as fp:
             print("load data from:", path)
             line = next(fp)
             line =line.strip()
             ss = line.split(" ")
             total = int(ss[0])
             dim = int(ss[1])
             assert (dim == expectDim),"dim:%d , expectDim: %d".format(dim,expectDim)
             ws = []
             mv = np.zeros(dim , dtype=np.float) #[0 for i in range(dim)]
             second = -1
             for t,line  in enumerate(fp):#@todo total
                 if ss[0] in [ '<UNK>',"<unk>"]:
                    second = t
                 line = line.strip()
                 ss = line.split(" ")
                 assert (len(ss) == (dim + 1))
                 vals = map(float , ss[1:])
                 #for i in range(1, dim + 1):
                 #    fv = float(ss[i])
                 #    mv[i - 1] += fv
                 #    vals.append(fv)
                 mv+= vals
                 ws.append(vals)
                 if len(ws)%50000==0:
                    print ("wordtvec data loading :",len(ws))
                 #if len(ws) >  50000 : 
                 #   break
             mv /= total
             #second = len(ws)
             while len(ws)%num_shards != 0:
                 ws.append(mv)
             #?~^?~J| ?~@个UNK?~M置

             #mv /=total
             assert (second != -1)
             # append one more token , maybe useless
             #ws.append(mv)
             if second != 1:
                 t = ws[1]
                 ws[1] = ws[second]
                 ws[second] = t
        print ("loading commpleted .....")
        print("make array 2d to 3d")
        total = len(ws)
        range_size = total / num_shards
        begin_ = 0
        ends_ = range_size
        ws = np.asarray(ws, dtype=np.float32)
        sub_ws= []
        for i in xrange(0 ,  num_shards ):
            begin_ = i*range_size
            if (i+1)*range_size < total :
                ends_ =  (i+1)*range_size
            else:
                ends_ = total
                assert ends_ - begin_ == range_size
            sub_ws.append( ws[ int(begin_)  : int(ends_) , ])
        return  np.array(sub_ws ,dtype = np.float32)

    def LoadModel(self, modelPath, vocabPath, maxSentenceLen, userDictPath):
        self.max_sentence_len_ = maxSentenceLen
        self.breaker_ = SentenceBreaker(maxSentenceLen);
        self.model = TfModel()
        self.model.restore(modelPath,"best_model-50000")
        self.w2v = self.load_w2v( FLAGS.num_shards , FLAGS.w2v_path, FLAGS.embedding_size)                                                        
        x_tensor = self.model.session.graph.get_tensor_by_name(name= FLAGS.transition_node_name + ":0")
        print "Reading from layer", x_tensor.name
	self.words =[]
	with tf.device("/cpu:0"):
	    for i in range( 0 , FLAGS.num_shards):
		 words_tensor_i = self.model.session.graph.get_tensor_by_name(name="words-%02d:0" % i)
                 self.words.append( words_tensor_i  )
	sess = self.model.session;
	
	flat_tensor = tf.reshape(x_tensor, [-1])  # 将多维的数据转换成一维
        count = sess.run(tf.size(flat_tensor))
        self.num_tags_ = int(math.sqrt(count))
        print "got num tag:", self.num_tags_
        self.transitions_ = sess.run(x_tensor)
        #print self.transitions_
        # load vocab
        self.vocab_ = load_vocab(vocabPath, self.vocab_)
        print " Total word: ", len(self.vocab_)
        self.scores_ = np.zeros([2, self.num_tags_], dtype=np.float32)
        self.bp_ = np.zeros([self.max_sentence_len_ , self.num_tags_], dtype=np.float32)
        # @todo add dict function

    def Segment(self, sentences, pTopResults):
        """
        切分句子
        :param sentences:
        :param pTopResults:
        :return:
        """
        assert (sentences is not None and len(sentences) > 0), "sentence can not be empty"
        input_tensor = self.model.session.graph.get_tensor_by_name(name=FLAGS.input_node_name + ":0")
        sess = self.model.session;
        input_tensor_mapped = np.zeros(shape=[len(sentences), self.max_sentence_len_], dtype=np.int32)
        for (k, words) in enumerate(sentences):
            len_words = len(words)
            print words.encode('utf8')
            if len_words <= 0:
                print "zero length str"
                return
            if len_words > self.max_sentence_len_:
                len_words = self.max_sentence_len_

            for i in range(0, len_words, 1):
                word = words[i]
		print word.encode("utf8") ,self.vocab_["<UNK>"]       
                if word in self.vocab_:
		    print k,i,self.vocab_[word] 
                    input_tensor_mapped[k][i] = self.vocab_[word]
                else:
                    input_tensor_mapped[k][i] = self.vocab_["<UNK>"]

            for i in range(len_words, self.max_sentence_len_, 1):
                input_tensor_mapped[k][i] = 0
       	    print "----- sentence tensor: {}".format(input_tensor_mapped[k])
        # tf.assign(input_tensor, input_tensor_mapped)
        pre_output_tensors = self.model.session.graph.get_tensor_by_name(name=FLAGS.scores_node_name + ":0")
        feed_dict_p = {input_tensor:input_tensor_mapped};
	for i in xrange(0, FLAGS.num_shards ):
           feed_dict_p[ self.words[ i ] ] = self.w2v[i];     
        predictions = sess.run(pre_output_tensors, feed_dict=feed_dict_p);
        for (k, words) in enumerate(sentences):
            len_words = len(words)
            if len_words <= 0:
                print "zero length str"
                return
            # @todo 需要增加自定义字典功能
            resultTags = list()
            self.scores_=np.zeros([2, self.num_tags_], dtype=np.float32)
            self.bp_ = np.zeros([ self.max_sentence_len_  , self.num_tags_], dtype=np.float32)
            self.bp_, self.scores_, resultTags = get_best_path(predictions, k, len_words, self.transitions_,
		 self.bp_,self.scores_, resultTags, self.num_tags_)
            #print  self.scores_
            #print  self.bp_
            assert len_words == len(resultTags), "num tag should equals setence len"
            resEle = list()
            start = 0
            for j in xrange(0, len_words, 1):
                if resultTags[len_words - j - 1] == 0:
                    if start < j:
                        resEle.append((start, j - start))
                    resEle.append((j, 1))
                    start = j + 1
                elif resultTags[len_words - j - 1] == 1:
                    if start < j:
                        resEle.append((start, j - start))
                    start = j
                elif resultTags[len_words - j - 1] == 2:
                    continue
                elif resultTags[len_words - j - 1] == 3:
                    resEle.append((start, j - start + 1))
                    start = j + 1
                else:
                    print "Unkonw tag:", resultTags[len_words - j - 1]
            if start < len_words:
                resEle.append((start, len_words - start))
            pTopResults.append(resEle)
        return pTopResults

    def SegmentpTags(self, sentence, pTopResults, pTags):
        """

        :param sentence:
        :param pTopResults:
        :param pTags:
        :return:
        """
        assert (sentence is not None and len(sentence.decode('utf8')) > 0), "sentence can not be empty"
        sentences = list()
        sentences = self.breaker_.breakSentences(sentence, sentences)
        if len(sentences) < 1:
            return None
        topResults = list()
        topResults = self.Segment(sentences, topResults)
        for (k, words) in enumerate(sentences):
            len_nn = len(topResults[k])
            todo = list()
            print  words.encode("utf8")
            print  topResults[k]
            for i in xrange(0, len_nn, 1):
                pTopResults.append(words[int(topResults[k][i][0]): int(topResults[k][i][0] + topResults[k][i][1])])
                todo.append(words[int(topResults[k][i][0]): int(topResults[k][i][0] + topResults[k][i][1])])
            # if ( pTags is not  None ) && self.
        return pTopResults


if __name__ == '__main__':
    dstr = "安装好tensorflow,切换到kcws代码目录";
    print dstr.decode('utf8')[0:3].encode('utf8')
    tfmodel_ = TfSegModel();
    tfmodel_.LoadModel(model_dir, basic_vocab, 80, None)
    pTopResults = list()
    pTopResults = tfmodel_.SegmentpTags("中华人民共和国", pTopResults, None)
    for seg in pTopResults:
        print seg


