#! /usr/bin/env python
#! coding=utf-8

from scipy.fftpack import dct
from settings import DCT_DIMENTION_NUM

class FeatureExtractor(object):
    """
    
    """
    def __init__(self):
        pass

    def dct(self, Array):
        """
        Discrete Cosine Transform
        """
        result = dct(Array, n = DCT_DIMENTION_NUM)
        return result

    #def do_extracte(self):
