#! /usr/bin/env python
#! coding=utf-8
import numpy

from scipy.fftpack import dct

from settings import DCT_DIMENTION_NUM
from decorators import param_length_matcher
from processor import SVMProcessor

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
        for V in ListV:
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

class SVMFeature(object):
    def __init__(self):
        self.features = []

    def append(self, feature):
        self.features.append(feature)
        
    def __str__(self):
        return ' '.join(self.features)

class SVMFeatureExtractor(FeatureExtractor):
    

    def __init__(self, ListVX, ListVY, ListVR):

        self.svmFeature = SVMFeature()

        # 将ListVX, ListVY, ListVR保存
        self.ListVX = ListVX
        self.ListVY = ListVY
        self.ListVR = ListVR

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

        self.stdVX = self._std(ListVX)
        self.stdVY = self._std(ListVY)

        ListAbsVX = SVMProcessor.abs_velocity_of_x(ListVX)
        ListAbsVY = SVMProcessor.abs_velocity_of_y(ListVY)
        self.avgAbsVX = self._avg(ListAbsVX)
        self.avgAbsVY = self._avg(ListAbsVY)

        self.stdAbsVX = self._std(ListAbsVX)
        self.stdAbsVY = self._std(ListAbsVY)

    def _points_lt_zero(self, ListA):
        cnt = 0
        for A in ListA:
            for a in A:
                if a < 0:
                    cnt += 1
        return cnt

    def _points_gt_zero(self, ListA):
        cnt = 0
        for A in ListA:
            for a in A:
                if a > 0:
                    cnt += 1
        return cnt

    def gen_features(self):
        """
            generate features
        """
        self.svmFeature.append(self.avgVX) # 1
        self.svmFeature.append(self.avgVY) # 2
        self.svmFeature.append(self.stdVX) # 3
        self.svmFeature.append(self.stdVY) # 4
        self.svmFeature.append(self.avgVR) # 5
        self.svmFeature.append(self.maxVX/self.maxVY) # 6
        self.svmFeature.append(self.avgPosVX/self.maxPosVX) # 7
        self.svmFeature.append(self.avgPosVY/self.maxPosVY) # 8
        self.svmFeature.append(self.avgNegVX/self.maxNegVX) # 9
        self.svmFeature.append(self.avgNegVY/self.maxNegVY) # 10
        self.svmFeature.append(self.avgVR/self.maxVR) # 11
        self.svmFeature.append(self.minVX/self.maxVX) # 12
        self.svmFeature.append(self.minVY/self.maxVY) # 13
        self.svmFeature.append(self.minVX/self.avgVX) # 14
        self.svmFeature.append(self.minVY/self.avgVY) # 15
        self.svmFeature.append(self.minVR/self.maxVR) # 16
        self.svmFeature.append(self.avgPosVX/self.avgNegVX) # 17
        self.svmFeature.append(self.avgPosVY/self.avgNegVY) # 18
        self.svmFeature.append(self.avgPosVX/self.avgPosVY) # 19
        self.svmFeature.append(self.avgPosVX/self.avgNegVY) # 20
        self.svmFeature.append(self.avgNegVX/self.avgPosVY) # 21
        self.svmFeature.append(self.avgNegVX/self.avgNegVY) # 22
        self.svmFeature.append(self.avgPosVX/self.maxNegVX) # 23
        self.svmFeature.append(self.avgPosVY/self.maxNegVY) # 24
        self.svmFeature.append(self.minVX/self.avgVX) # 25
        self.svmFeature.append(self.minVY/self.avgVY) # 26
        self.svmFeature.append(self.avgVX/self.maxVX) # 27
        self.svmFeature.append(self.avgVY/self.maxVY) # 28
        self.svmFeature.append(self.avgPosVX) # 29
        self.svmFeature.append(self.avgPosVY) # 30
        self.svmFeature.append(self.avgNegVX) # 31
        self.svmFeature.append(self.avgNegVY) # 32
        self.svmFeature.append(self.maxVX) # 33
        self.svmFeature.append(self.maxVY) # 34
        self.svmFeature.append(self.avgVX) # 35
        self.svmFeature.append(self.avgVY) # 36
        self.svmFeature.append(self.avgAbsVX) # 37
        self.svmFeature.append(self.avgAbsVY) # 38
        self.svmFeature.append(self.stdAbsVX) # 39
        self.svmFeature.append(self.stdAbsVY) # 40
        self.svmFeature.append(self._points_lt_zero(self.ListVX)) # 41
        self.svmFeature.append(self._points_lt_zero(self.ListVY)) # 42
        self.svmFeature.append(self._points_gt_zero(self.ListVX)) # 43
        self.svmFeature.append(self._points_gt_zero(self.ListVY)) # 44
        self.svmFeature.append(self.maxNegVX) # 45
        self.svmFeature.append(self.maxNegVY) # 46
