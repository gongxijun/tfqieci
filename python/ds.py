#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     ds.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/20
-----------------------------------
Change Activiy  :   2018/11/20
-----------------------------------

"""
import tensorflow as tf
import numpy as np

# sess=tf.Session()

# make a np array that is 1.1 GB
size_in_gb = 3.1  # so that 2 of them will exceed 2GB
float32_bytes = 4
num_shards = 3

nelements = int(size_in_gb * float(1 << 30) / (float32_bytes*num_shards*200))
foo = np.ones([num_shards ,nelements,200], dtype=np.float32)
#embeddings = tf.Variable(tf.random_uniform(foo.shape, minval=-0.1, maxval=0.1), trainable=False)

# convert np array to tensors (NOTE: this implicitly creates tf.constant's)
# t2 = embeddings.assign(foo)
# This bakes the graphdef and throws an error when it exceeds 2GB
# sess.run(embeddings.initializer, feed_dict={embeddings.initial_value:foo})

#input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
weights = []
for i in range(0, num_shards):
    weights_i = tf.get_variable("words-%02d" % i,
                                initializer=tf.random_uniform(foo[i].shape, minval=-0.1, maxval=0.1), trainable=False)
    weights.append(weights_i)

input_ids = tf.transpose(tf.stack([[1, 2], [3, 0], [8, 2], [5, 1]]))
input_embedding = tf.nn.embedding_lookup(weights, input_ids, partition_strategy="div")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print weights
#print(sess.run(weights ,feed_dict={weight_i: var for weight_i, var  in zip(weights,foo)}))

print(sess.run(input_embedding, feed_dict={weight_i: var for weight_i, var  in zip(weights,foo)}))
