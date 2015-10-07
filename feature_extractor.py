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

    def features_of_velocity(self, ListVX, ListVY, ListVR):
        """
            velocity related features
        """
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

        ListAbsVX = SVMProcessor.abs_velocity_of_x(SVMProcessor, ListVX)
        ListAbsVY = SVMProcessor.abs_velocity_of_y(SVMProcessor, ListVY)
        self.avgAbsVX = self._avg(ListAbsVX)
        self.avgAbsVY = self._avg(ListAbsVY)

        self.stdAbsVX = self._std(ListAbsVX)
        self.stdAbsVY = self._std(ListAbsVY)

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

    def features_of_time(self, ListT, ListVX, ListVY):

        """
            time related features
        """
        self.penUpTimes = len(ListT)
        self.totalTime = ListT[-1][-1] - ListT[0][0]
        self.maxPenDownTime = -1
        self.minPenDownTime = 99999
        self.totalPenDownTime = 0
        self.totalPenUpTime = 0
        for T in ListT:
            penDownTime = T[-1] - T[0]
            self.maxPenDownTime = max(self.maxPenDownTime, penDownTime)
            self.minPenDownTime = min(self.minPenDownTime, penDownTime)
            self.totalPenDownTime += penDownTime
        self.totalPenUpTime = self.totalTime - self.totalPenDownTime
        self.totalPosVXTime = 0
        self.totalNegVXTime = 0
        self.totalPosVYTime = 0
        self.totalNegVYTime = 0
        for VX in ListVX:
            for vx in VX:
                if vx > 0:
                    self.totalPosVXTime += 10
                elif vx < 0:
                    self.totalNegVXTime += 10
        for VY in ListVY:
            for vy in VY:
                if vy > 0:
                    self.totalPosVYTime += 10
                elif vy < 0:
                    self.totalNegVYTime += 10

        self.svmFeature.append(self.penUpTimes) # 47
        self.svmFeature.append(self.totalTime) # 48
        self.svmFeature.append(self.maxPenDownTime / self.totalTime) # 49
        self.svmFeature.append(self.minPenDownTime / self.totalTime) # 50
        self.svmFeature.append(self.totalPenDownTime / self.totalTime) # 51
        self.svmFeature.append(self.totalPenUpTime / self.totalTime) # 52
        self.svmFeature.append(self.totalPenDownTime) # 53
        self.svmFeature.append(self.totalPosVXTime / self.totalPosVYTime) # 54
        self.svmFeature.append(self.totalNegVYTime / self.totalNegVYTime) # 55
        self.svmFeature.append(self.totalPosVXTime / self.totalTime) # 56
        self.svmFeature.append(self.totalPosVYTime / self.totalTime) # 57
        self.svmFeature.append(self.totalNegVXTime / self.totalTime) # 58
        self.svmFeature.append(self.totalNegVYTime / self.totalTime) # 59

    def features_of_shape(self, ListX, ListY, ListR):
        """
            shape related features
        """
        maxX = self._max(ListX)
        minX = self._min(ListX)
        maxY = self._max(ListY)
        minY = self._min(ListY)

        shiftX = 0
        shiftY = 0
        shiftR = 0
        shiftPosX = 0
        shiftPosY = 0
        for X in ListX:
            for i in range(len(X) - 1):
                delta = X[i+1] - X[i]
                shiftX += abs(delta)
                if delta > 0:
                    shiftPosX += delta

        for Y in ListY:
            for i in range(len(Y) - 1):
                delta += Y[i+1] - Y[i]
                shiftY += abs(delta)
                if delta > 0:
                    shiftPosY += delta

        for R in ListR:
            for r in R:
                shiftR += r

        totalPointsCnt = 0
        pointsCntIn1stQ = 0
        pointsCntIn2ndQ = 0
        pointsCntIn3rdQ = 0
        pointsCntIn4thQ = 0

        for X in ListX:
            totalPointsCnt += len(X)
        for i in range(len(ListX)):
            for j in range(len(ListX[i])):
                if ListX[i][j] >= 0:
                    if ListY[i][j] >= 0:
                        pointsCntIn1stQ += 1
                    else:
                        pointsCntIn4thQ += 1
                else:
                    if ListY[i][j] >= 0:
                        pointsCntIn2ndQ += 1
                    else:
                        pointsCntIn3rdQ += 1

        self.svmFeature.append((maxX - minX) / (maxY - minY)) # 60
        self.svmFeature.append(shiftX / shiftR) # 61
        self.svmFeature.append(shiftY / shiftR) # 62
        self.svmFeature.append((maxX - minX) * (maxY - minY)) # 63
        self.svmFeature.append(shiftR) # 64
        self.svmFeature.append(pointsCntIn1stQ) # 65
        self.svmFeature.append(pointsCntIn2ndQ) # 66
        self.svmFeature.append(pointsCntIn3rdQ) # 67
        self.svmFeature.append(pointsCntIn4thQ) # 68
        self.svmFeature.append(0) # 69 ######TODO#######
        self.svmFeature.append(shiftR / ((maxX - minX) * (maxY - minY))) # 70
        self.svmFeature.append(shiftPosX / shiftX) # 71
        self.svmFeature.append(shiftPosY / shiftY) # 72

    def features_of_pression(self, ListP, ListVP):
        """
            pression related features
        """
        avgP = self._avg(ListP)
        minP = self._min(ListP)
        maxP = self._max(ListP)
        stdP = self._std(ListP)
        stdVP = self._std(ListVP)
        cntGtAvg = 0
        cntLtAvg = 0
        for P in ListP:
            for p in P:
                if p > avgP:
                    cntGtAvg += 1
                else:
                    cntLtAvg += 1
        
        self.svmFeature.append(avgP) # 73
        self.svmFeature.append(minP) # 74
        self.svmFeature.append(stdP) # 75
        self.svmFeature.append(stdVP) # 76
        self.svmFeature.append(avgP / maxP) # 77
        self.svmFeature.append(cntGtAvg) # 78
        self.svmFeature.append(cntLtAvg) # 79

    def features_of_zero(self, ListVX, ListVY, ListAX, ListAY):
        """
            features about zero
        """
        totalCnt = 0.0
        for VX in ListVX:
            totalCnt += len(VX)
        pointsCntZeroVX = 0
        pointsCntZeroVY = 0
        pointsCntZeroAX = 0
        pointsCntZeroAY = 0
        for VX in ListVX:
            for i in range(len(VX) - 1):
                if VX[i]*VX[i+1] < 0 or VX[i] == 0:
                    pointsCntZeroVX += 1
        for VY in ListVY:
            for i in range(len(VY) - 1):
                if VY[i]*VY[i+1] < 0 or VY[i] == 0:
                    pointsCntZeroVY += 1
        
        for AX in ListAX:
            for i in range(len(AX) - 1):
                if AX[i]*VX[i+1] < 0 or AX[i] == 0:
                    pointsCntZeroAX += 1
        for AY in ListAY:
            for i in range(len(AY) - 1):
                if AY[i]*AY[i+1] < 0 or AY[i] == 0:
                    pointsCntZeroAY += 1
        
        self.svmFeature.append(pointsCntZeroVX / totalCnt) # 80
        self.svmFeature.append(pointsCntZeroVY / totalCnt) # 81
        self.svmFeature.append(pointsCntZeroAX / totalCnt) # 82
        self.svmFeature.append(pointsCntZeroAY / totalCnt) # 83
        self.svmFeature.append(0) # 84 ####TODO####
        self.svmFeature.append(0) # 85 ####TODO####
        self.svmFeature.append(0) # 86 ####TODO####
        self.svmFeature.append(0) # 87 ####TODO####
        

    def features_of_acceleration(self, ListAR, ListAX, ListAY):
        """
        acceleration related features
        """
        maxAR = self._max(ListAR)
        maxAX = self._max(ListAX)
        maxAY = self._max(ListAY)
        minAR = self._min(ListAR)
        minAX = self._min(ListAX)
        minAY = self._min(ListAY)
        avgAR = self._avg(ListAR)
        avgAX = self._avg(ListAX)
        avgAY = self._avg(ListAY)
        stdAX = self._std(ListAX)
        stdAY = self._std(ListAY)
        avgPosAX = numpy.mean(self._select_positive(ListAX))
        avgPosAY = numpy.mean(self._select_positive(ListAY))
        avgNegAX = numpy.nean(self._select_negative(ListAX))
        avgNegAY = numpy.mean(self._select_negative(ListAY))
        ListAbsAX = SVMProcessor.abs_acc_of_x(SVMProcessor, ListAX)
        ListAbsAY = SVMProcessor.abs_acc_of_y(SVMProcessor, ListAY)
        avgAbsAX = self._avg(ListAbsAX)
        avgAbsAY = self._avg(ListAbsAY)
        maxAbsAX = self._max(ListAbsAX)
        maxAbsAY = self._max(ListAbsAY)
        stdAbsAX = self._std(ListAbsAX)
        stdAbsAY = self._std(ListAbsAY)
        pointsCntGtZeroAX = self._points_gt_zero(ListAX)
        pointsCntLtZeroAX = self._points_lt_zero(ListAX)
        pointsCntGtZeroAY = self._points_gt_zero(ListAY)
        pointsCntLtZeroAY = self._points_lt_zero(ListAY)
        
        self.svmFeature.append(maxAR) # 88
        self.svmFeature.append(avgAR) # 89
        self.svmFeature.append(0) # 90 ####TODO####
        self.svmFeature.append(0) # 91 ####TODO####
        self.svmFeature.append(stdAX) # 92
        self.svmFeature.append(stdAY) # 93
        self.svmFeature.append(avgAbsAX / maxAbsAX) # 94
        self.svmFeature.append(avgAbsAY / maxAbsAY) # 95
        self.svmFeature.append(avgAX / self.avgPosVX) # 96
        self.svmFeature.append(avgAY / self.avgPosVY) # 97
        self.svmFeature.append(avgAR / self.maxVR) # 98
        self.svmFeature.append(minAR / self.avgVR) # 99
        self.svmFeature.append(minAY / avgAY) # 100
        self.svmFeature.append(avgPosAX / avgNegAX) # 101
        self.svmFeature.append(maxAR / self.avgPosVX) # 102
        self.svmFeature.append(maxAR / self.avgPosVY) # 103
        self.svmFeature.append(maxAR / self.avgNegVX) # 104
        self.svmFeature.append(maxAR / self.avgNegVY) # 105
        self.svmFeature.append(minAR / self.avgPosVX) # 106
        self.svmFeature.append(minAR / self.avgPosVY) # 107
        self.svmFeature.append(minAR / self.avgNegVX) # 108
        self.svmFeature.append(minAR / self.avgNegVY) # 109
        self.svmFeature.append(avgNegAY / avgPosAY) # 110
        self.svmFeature.append(avgAR / self.avgNegVX) # 111
        self.svmFeature.append(avgAR / self.avgNegVY) # 112
        self.svmFeature.append(avgAbsAX) # 113
        self.svmFeature.append(avgAbsAY) # 114
        self.svmFeature.append(stdAbsAX) # 115
        self.svmFeature.append(stdAbsAY) # 116
        self.svmFeature.append(maxAX) # 117
        self.svmFeature.append(minAX) # 118
        self.svmFeature.append(maxAY) # 119
        self.svmFeature.append(minAY) # 120
        self.svmFeature.append(avgAX) # 121
        self.svmFeature.append(avgAY) # 122
        self.svmFeature.append(pointsCntGtZeroAX) # 123
        self.svmFeature.append(pointsCntLtZeroAX) # 124
        self.svmFeature.append(pointsCntGtZeroAY) # 125
        self.svmFeature.append(pointsCntLtZeroAY) # 126

    def features_of_curve(self, ListX, ListY):
        """
            signature curve related features
        """
        avgX = self._avg(ListX)
        avgY = self._avg(ListY)
        stdX = self._std(ListX)
        stdY = self._std(ListY)

        self.svmFeature.append(avgX) # 127
        self.svmFeature.append(avgY) # 128
        self.svmFeature.append(stdX) # 129
        self.svmFeature.append(stdY) # 130

    def generate_features(self, ListT, ListX, ListY, ListR, ListP,
            ListVX, ListVY, ListVR, ListVP, ListAX, ListAY, ListAR):
        """
            driver to generate features
        """
        self.features_of_velocity(ListVX, ListVY, ListVR)
        self.features_of_time(ListT, ListVX, ListVY)
        self.features_of_shape(ListX, ListY, ListR)
        self.features_of_pression(ListP, ListVP)
        self.features_of_zero(ListVX, ListVY, ListAX, ListAY)
        self.features_of_acceleration(ListAR, ListAX, ListAY)
        self.features_of_curve(ListX, ListY)


