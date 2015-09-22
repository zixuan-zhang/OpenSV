#! /usr/bin/env python
#coding=utf-8

"""
@Date:   2015年09月22日
@Author: Zhang Zixuan
@Blog:   www.noathinker.com
@Des:    pre-processing
"""

import numpy

class Processor(object):
    """
    本类提供了预处理的一些函数
    """

    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y

    def location_normalization(self, X, Y):
        """
        将位置归一化到原点
        """
        meanX = numpy.mean(X)
        meanY = numpy.mean(Y)

        resultX = [x - meanX for x in X]
        resultY = [y - meanY for y in Y]
        return resultX, resultY

    def size_normalization(self, X, Y):
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

        mX = 400
        mY = 100

        resultX = [mX * (x - minX) / rangeX for x in X]
        resultY = [mY * (y - minY) / rangeY for y in Y]
        return resultX, resultY

    def gauss_smoothing(self, params):
        """
        使用高斯滤波平滑数据
        """
        fenmu = sum([numpy.exp(-1*pow(j,2)/2.0) for j in range(-2, 2)])
        result = []
        for param in params:
            resultParam = [sum([numpy.exp(-1*pow(i,2)/2.0)/fenmu*x for i in range(-2, 2)]) for x in param]
            result.append(resultParam)
        return result

if __name__ == "__main__":
    processor = Processor()
    X = [[1, 1.2, 1.3, 1.1]]
