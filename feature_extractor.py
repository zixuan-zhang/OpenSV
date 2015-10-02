#! /usr/bin/env python
#! coding=utf-8
import numpy

from scipy.fftpack import dct

from settings import DCT_DIMENTION_NUM
from decorators import param_length_matcher

class DCTFeatureExtractor(object):
    def __init__(self):
        pass

    def dct(self, Array, n = DCT_DIMENTION_NUM):
        """
        generate DCT coefficients
        """
        if not Array:
            return []

        result = []
        N = len(Array)
        for i in range(n):
            c = 1.0 if i == 0 else 2.0
            res = sum([numpy.sqrt(c/N)*Array[k]*numpy.cos(numpy.pi*(2*k+1)*i/2/N) for k in range(N)])
            result.append(res)

        return result

class FeatureExtractor(object):
    """
     
    """
    def __init__(self):
        pass

    def _avg(self, ListV):
        """
            inner function
            求ListV中数值的平均
        """
        _sum = 0.0
        _cnt = 0
        for V in ListVX:
            _cnt += len(V)
            _sum += sum(V)
        avg = _sum / _cnt
        return avg

    def _max(self, ListA):
        __max = -99999.99
        for A in ListA:
            __max = max(max(A), __max)
        return __max

    def _min(self, ListA):
        __min = 999999999.99
        for A in ListA:
            __min = min(min(A), __min)
        return __min

    def _std(self, ListA):
        A = []
        for _A in ListA:
            for a in _A:
                A.append(a)
        stdA = numpy.std(A)
        return stdA

    """
        _select_* 函数好像不好使
    """
    def _select_positive(self, ListA):
        """
        """
        posA = []
        for A in ListA:
            for a in A:
                if a >= 0:
                    posA.append(a)
        return posA

    def _select_negative(self, ListA):
        negA = []
        for A in ListA:
            for a in A:
                if a < 0:
                    negA.append(a)
        return negA

class SVMFeatureExtractor(FeatureExtractor):
    
    class SVMFeature(object):
        def __init__(self):

    def __init__(self, ListVX, ListVY, ListVR):
        self.avgVX = self._avg(ListVX)
        self.avgVY = self._avg(ListVY)
        self.avgVR = self._avg(ListVR)

        self.maxVX = self._max(ListVX)
        self.maxVY = self._max(ListVY)
        self.maxVR = self._max(ListVR)

        self.minVX = self._min(ListVX)
        self.minVY = self._min(ListVY)
        self.minVR = self._min(ListVR)

        posVX = self._select_positive(ListVX)
        posVY = self._select_positive(ListVY)
        negVX = self._select_negative(ListVX)
        negVY = self._select_negative(ListVY)

        self.avgPosVX = numpy.mean(posVX)
        self.avgPosVY = numpy.mean(posVY)

        self.maxPosVX = max(posVX)
        self.maxPosVY = max(posVY)

        self.minPosVX = min(posVX)
        self.minPosVY = min(posVY)

        self.avgNegVX = numpy.mean(negVX)
        self.avgNegVY = numpy.mean(negVY)

        self.maxNegVX = max(negVX)
        self.maxNegVY = max(negVY)

        self.minNegVX = min(negVX)
        self.minNegVY = min(negVY)
