#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     demo.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/16
-----------------------------------
Change Activiy  :   2018/11/16
-----------------------------------

"""
import numpy as np
import tensorflow as tf

__author__ = 'xijun1'

# tf.reset_default_graph()
# graph = tf.Graph()
# with graph.as_default():
#     input_x=tf.placeholder(tf.int32,shape=[2,2])
#     embeddings_matrix = np.zeros(shape=[2,2],dtype=np.float)
#     with tf.Session() as sess:
#         embeddings =tf.Variable(input_x,name="words")
#         #embedding_loopup = tf.nn.embedding_lookup(embeddings, input_x)
#         sess.run(tf.global_variables_initializer(), feed_dict={input_x: [[1, 2], [3, 4]]})
#         print(sess.run(embeddings))
#
#
# tf.reset_default_graph()
# graph = tf.Graph()
# with graph.as_default():
#     input_x=tf.placeholder(tf.int32,shape=[2,2])
#     embeddings_matrix = np.zeros(shape=[2, 2], dtype=np.float)
#     embeddings = tf.Variable(input_x, name="words")
#     #c= tf.add(embeddings,embeddings)
#     tf.get_variable
#     # print(sess.run(embeddings,feed_dict={input_x: [[1, 2], [3, 4]]}))
#     sv = tf.train.Supervisor(graph=graph,summary_op= None, logdir="./logs")
#     tf.train.replica_device_setter()
#     # embeddings_matrix = np.zeros(shape=[2,2],dtype=np.float)
#     with sv.managed_session() as sess:
#         sv.loop(60, lambda: sv.summary_computed(sess, sess.run(tf.global_variables_initializer(),feed_dict={input_x: [[1, 2], [3, 4]]})))
#         #print sess.run(embeddings)
#         #embedding_loopup = tf.nn.embedding_lookup(embeddings, embeddings)
#         #print(sess.run(embeddings))
#
#
#

# tf.nn.embedding_lookup
# num_shards = 3
# a = np.arange(30).reshape(10, 3)
# range_len = a.shape[0] / num_shards
# b = a.shape[1]
# print b
# list_s = []
# ids = []
# for i in range(1, num_shards):
#     list_s.append(a[(i - 1) * range_len: i * range_len, ])
#     ids.append(np.arange((i - 1) * range_len, i * range_len))
# list_s.append(a[(num_shards - 1) * range_len:b, ])
# a = None
# print ids
# print list_s
#
# print int(10 / 3)

# !/usr/bin/env/python
# coding=utf-8
import tensorflow as tf
import numpy as np

input_ids = tf.placeholder(dtype=tf.int32, shape=[None,None])

num_shards = 3
weights = []
weights_shape = np.arange(27).reshape(9, 3)
# assert weights_shape[0] % num_shards == 0
num_shards_len = (weights_shape.shape[0]) / num_shards
assert  (weights_shape.shape[0]) % num_shards ==0
begin_ = 0
ends_ = num_shards_len
for i in range(0, num_shards):
    if (i + 1) * num_shards_len < weights_shape.shape[0]:
        begin_ = i * num_shards_len
        if i + 1 == num_shards:
            ends_ = weights_shape.shape[0]
        else:
            ends_ = (i + 1) * num_shards_len
    else:
        begin_ = i * num_shards_len
        ends_ = weights_shape.shape[0]
    weights_i = tf.get_variable("words-%02d" % i,
                                initializer=(weights_shape[begin_: ends_, ]))
    weights.append(weights_i)

input_embedding = tf.nn.embedding_lookup(weights, input_ids,partition_strategy="div")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(weights))

print(sess.run(input_embedding, feed_dict={input_ids: [[1, 2], [3, 0], [8, 2], [5, 1]]}))
