#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
----------------------------------
Version    : ??
File Name :     viterbi_decode.py
Description :   
Author  :       xijun1
Email   :       xijun1@staff.weibo.com
Date    :       2018/11/15
-----------------------------------
Change Activiy  :   2018/11/15
-----------------------------------

"""
__author__ = 'xijun1'

import tensorflow as tf


def viterbi_decode(predictions, sentenceIdx, nn, trans, bp, scores, ntag):
    """
    :param predictions:
    :param sentenceIdx:
    :param nn:
    :param trans:
    :param bp:
    :param scores:
    :param ntag:
    :return:
    """
    for tag in xrange(0, ntag, 1):
        scores[0][ tag ] = predictions[sentenceIdx][0][tag]
    for i in xrange(0, nn, 1):
        for t in xrange(0, ntag, 1):
            maxScore = -1e7
            emission = predictions[sentenceIdx][i][t]
            for prev in xrange(0, ntag, 1):
                score = scores[(i - 1) % 2][prev] + trans[prev][t] + emission
                if score > maxScore:
                    maxScore = score
                    bp[i - 1][t] = prev
            scores[i % 2][t] = maxScore
    maxScore = scores[(nn - 1) % 2][0]
    ret = 0
    for i in xrange(0, ntag, 1):
        if scores[(nn - 1) % 2][i] > maxScore:
            ret = i
            maxScore = scores[(nn - 1) % 2][i]
    return ret, bp, scores


def get_best_path(predictions, sentenceIdx, nn, trans, bp, scores, resultTags, ntags):
    """

    :param predictions:
    :param sentenceIdx:
    :param nn:
    :param trans:
    :param bp:
    :param scores:
    :param resultTags:
    :param ntags:
    :return:
    """
    lastTag, bp, scores = viterbi_decode(predictions, sentenceIdx, nn, trans, bp, scores, ntags)
    resultTags.append(lastTag)
    for i in reversed(range(0, nn - 1)):  # [n-2,n-3,....2,1,0]
        bpTag = bp[i][int(lastTag)]
        resultTags.append(bpTag)
        lastTag = bpTag
    return bp, scores, resultTags
