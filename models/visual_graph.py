#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     seg_demo.py
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
import os
model_dir = '/data1/xijun1/tfqieci/models/seg_model.pbtxt'
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph(model_dir)
for op in graph.get_operations():
    print(op.name)
