#! /usr/bin/env python
#coding=utf-8

"""
@Date:   2015年09月22日
@Author: Zhang Zixuan
@Blog:   www.noathinker.com
@Des:    pre-processing
"""

import numpy

class PreProcessor(object):
    """
        本类提供了数据处理的一些函数
    """

    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y

    """
        数据预处理部分：
            @ location_normalization
            @ size_normalization
            @ gauss_smoothing
            @ strip_zero
            @ duplicated_point_split
    """

    def location_normalization(self, X, Y):
        """
            将位置归一化到原点
        """
        meanX = numpy.mean(X)
        meanY = numpy.mean(Y)

        resultX = [x - meanX for x in X]
        resultY = [y - meanY for y in Y]
        return resultX, resultY

    def offset_to_origin_normalization(self, X, Y):
        """
            使坐标为原始坐标的offset
        """
        if len(X) == 0:
            return X, Y
        x0 = X[0]
        y0 = Y[0]
        XR = [xi - x0 for xi in X]
        YR = [yi - y0 for yi in Y]
        return XR, YR

    def size_normalization(self, X, Y, width, height):
        """
            轨迹大小归一化
            将轨迹缩放到400 * 100的空间内
            假定 x, y均为正值
        """
        minX = numpy.min(X)
        minY = numpy.min(Y)
        maxX = numpy.max(X)
        maxY = numpy.max(Y)

        rangeX = maxX - minX
        rangeY = maxY - minY

        # mX = 400
        # mY = 100
        mX = width
        mY = height

        resultX = [float(mX) * (x - minX) / rangeX for x in X]
        resultY = [float(mY) * (y - minY) / rangeY for y in Y]
        return resultX, resultY

    def gauss_smoothing(self, *params):
        """
            使用高斯滤波平滑数据
        """
        fenmu = sum([numpy.exp(-1*pow(j,2)/2.0) for j in range(-2, 2)])
        result = []
        for param in params:
            resultParam = [sum([numpy.exp(-1*pow(i,2)/2.0)/fenmu*x for i in range(-2, 2)]) for x in param]
            result.append(resultParam)
        return result

    def strip_zero(self, T, X, Y, P=None):
        """
            消除零点。
        """
        pass

    def duplicated_point_split(self, *args, **kwargs):
        """
            过滤掉重复时间戳的点，默认删除后面的重复点
            时间戳序列存放在kwargs中，其他序列存放在args中
            思路：
                1. 记录重复的时间戳的index
                2. 将args中和T中分别删掉该index对应的item
                3. 注意每删掉一次，后面的index将会减少1
        """
        if kwargs.has_key('T'):
            indexes = []
            T = kwargs['T']
            for i in range(len(T) - 1):
                if T[i] == T[i+1]:
                    indexes.append(i+1)
            for i in range(len(indexes)):
                for param in args:
                    param.pop(indexes[i] - i)
                T.pop(indexes[i] - i)

    def signature_segmentation(self, T, X, Y, P):
        """
            将数据根据零点进行分段
        """
        indexes = [0]
        for i in range(len(T) - 1):
            if (T[i+1] - T[i]) > 10:
                indexes.append(i+1)
        indexes.append(len(T)-1)
        segmentsT = []
        segmentsX = []
        segmentsY = []
        segmentsP = []
        for i in range(len(indexes)-1):
            if (indexes[i+1] - indexes[i]) == 1:
                continue
            segmentsT.append(T[indexes[i]:indexes[i+1]])
            segmentsX.append(X[indexes[i]:indexes[i+1]])
            segmentsY.append(Y[indexes[i]:indexes[i+1]])
            segmentsP.append(P[indexes[i]:indexes[i+1]])
        return segmentsT, segmentsX, segmentsY, segmentsP
