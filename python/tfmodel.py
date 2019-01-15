#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     tfmodel.py
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
import os , sys

class TfModel:
    def __init__(self):
        self.sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions())
        self.session = None;
        self.x = None;
        self.y = None;
        pass

    def load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="")
            self.session = tf.Session(graph=graph, config=self.sess_config)
        return graph
    
    def restore(self, checkpoint_dir, latest_filename='best_model'):
        """

        :param checkpoint_dir:  模型保存的路径
        :param latest_filename: 模型前缀
        :return:  返回模型的图
        """
        self.session = tf.Session(config=self.sess_config)
        model_meta_path = os.path.join(checkpoint_dir, latest_filename)+".meta"
        print("graph path: ",model_meta_path)
        saver = tf.train.import_meta_graph( model_meta_path)
        module_file =tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir,latest_filename=latest_filename)
        if module_file is None:
           print("init module_file ....")
	   module_file = os.path.join(checkpoint_dir, latest_filename)
	print("module path: ", module_file)
        saver.restore(self.session, module_file)
        return self.session.graph

    def Eval(self, inputNames, outputNames, data):
        if inputNames == None:
            self.x =None
        else:
            self.x = self.session.graph.get_tensor_by_name(inputNames + ":0")
        self.y = self.session.graph.get_tensor_by_name(outputNames + ":0")
        return self.session.run(self.y, feed_dict={self.x: data});

    def Eval(self, inputTensor, outputTensor, data):
        return self.session.run(outputTensor, feed_dict={inputTensor: data});

