#! /usr/bin/env python
#coding=utf-8

"""
@Date:   2015年09月22日
@Author: Zhang Zixuan
@Blog:   www.noathinker.com
@Des:    pre-processing
"""

import numpy

from exception import ParamErrorException
from decorators import param_length_matcher

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

class DataProcessor(object):

    """
    """

    def __init__(self):
        pass

    def _sqrt(self, A, B):
        """
            平方根局部函数
        """
        C = [numpy.sqrt(pow(A[i], 2) + pow(B[i], 2)) for i in range(len(A))]
        return C

    def _derivative(self, A, T):
        """
            求导数子函数
        """
        D = [(A[i+1]-A[i])/(T[i+1]-T[i]) for i in range(len(A)-1)]
        D.appen(D[-1]) #为了保证D的结果与A和T长度一致
        return D

    def _divisor(self, A, B):
        """
            除法子函数
        """
        D = [A[i] / B[i] for i in range(len(A))]
        return D

    @param_length_matcher
    def radius(self, X, Y):
        """
            @num   : 3
            @des   : calculate radius signal
            @input : X for x axis # list
                     Y for y axis # list
            @output: R for radius # list
        """
        return self._sqrt(X, Y)

    @param_length_matcher
    def velocity_of_x(self, X, T):
        """
            @num   : 4
            @des: calculate velocity signal
            @input : A for array, could be x,y axis or radius
                   : T for timestamp
            @output: V for velocity
        """
        VX = self._derivative(X, T)
        return VX

    @param_length_matcher
    def velocity_of_y(self, Y, T):
        """
            @num   : 5
        """
        VY = self._derivative(Y, T)
        return VY

    @param_length_matcher
    def abs_velocity(self, VX, VY):
        """
            @num   : 6
            @des   : calculate absolute velocity
            @input : VX for x axis velocity 
                   : VY for y axis velocity
            @output: AV for absolute velocity
        """
        AV = self._sqrt(VX, VY)
        return AV

    @param_length_matcher
    def velocity_of_r(self, R, T):
        """
            @num   : 7
        """
        VR = self._derivative(R, T)

    @param_length_matcher
    def angle_of_velocity(self, VY, VX):
        """
            @num   : 8
        """
        pass

    @param_length_matcher
    def sin_of_angle_v(self, VY, VR):
        """
            @num   : 9
            @des   : sinuous of angle, sin
            @input : VX for x axis velocity
                     VY for y axis velocity
            @output: sinV
        """
        sinV = self._divisor(VY, VR)
        return sinV

    @param_length_matcher
    def cos_of_angle_v(self, VX, VR):
        """
            @num   : 10
            @des   : cosine of angle, cos
            @input : VX for x axis velocity
                     VY for y axis velocity
            @outpu:  cosV
        """
        cosV = self._divisor(VX, VR)
        return cosV

    @param_length_matcher
    def acc_of_vx(self, VX, T):
        """
            @num   : 11
        """
        return self._derivative(VX, T)

    @param_length_matcher
    def acc_of_vy(self, VY, T):
        """
            @num   : 12
        """
        return self._derivative(VY, T)
        
    @param_length_matcher
    def abs_acc(self, AX, AY):
        """
            @num   : 13
        """
        return self._sqrt(AX, AY)

    @param_length_matcher
    def tan_acc(self, VX, AX, VY, AY, VR):
        """
            @num   : 14
        """
        TA = [(VX[i]*AX[i]+VY[i]*AY[i])/VR[i] for i in range(VX)]
        return TA
        
    @param_length_matcher
    def cen_acc(self, VX, AX, VY, AY, VR):
        """
            @num   : 15
        """
        CA = [(VX[i]*AY[i]-VY[i]*AX[i])/VR[i] for i in range(VX)]

    @param_length_matcher
    def acc_of_radius(self, VR, T):
        """
            @num   : 16
        """
        AR = self._derivative(VR, T)
        return AR

    @param_length_matcher
    def angle_of_acc(self, AY, AX):
        """
            @num   : 17
        """
        pass

    @param_length_matcher
    def sin_of_angle_a(self, AY, AR):
        """
            @num   : 18
        """
        sinOfAngle = self._derivative(AY, AR)
        return sinOfAngle

    @param_length_matcher
    def cos_of_angle_a(self, AX, AR):
        """
            @num   : 19
        """
        cosOfAngle = self._derivative(AX, AR)
        return cosOfAngle

    @param_length_matcher
    def angle_of_centripetal_acc(self, AY, AX):
        """
            @num   : 20
        """
        pass
    
    @param_length_matcher
    def sin_of_angle_b(self, AC, AR):
        """
            @num   : 21
        """
        pass

    @param_length_matcher
    def cos_of_angle_b(self, AT, AR):
        """
            @num   : 22
        """
        pass

    @param_length_matcher
    def jerk_in_x(self, AX, T):
        """
            @num   : 23
        """
        JX = self._derivative(AX, T)
        return JX

    @param_length_matcher
    def jerk_in_y(self, AY, T):
        """
            @num   : 24
        """
        JY = self._derivative(AY, T)
        return JY

    @param_length_matcher
    def abs_jerk(self, JX, JY):
        """
            @num   : 25
        """
        absJerk = self._sqrt(JX, JY)
        return absJerk

    @param_length_matcher
    def tan_jerk(self, AX, JX, AY, JY, AR):
        """
            @num   : 26
        """
        TJ = [(AX[i]*JX[i]+AY[i]*JY[i])/AR[i] for i in range(len(AR))]
        return TJ
    
    @param_length_matcher
    def cent_jerk(self, AX, JX, AY, JY, AR):
        """
            @num   : 27
        """
        CJ = [(AX[i]*JY[i]-AY[i]*JX[i])/AR[i] for i in range(len(AR))]
        return CJ

    @param_length_matcher
    def jerk_in_r(self, AR, T):
        """
            @num   : 28
        """
        JR = self._derivative(AR, T)
        return JR
            
    @param_length_matcher
    def angle_of_jerk(self, JX, JY):
        """
            @num   : 29
        """
        pass

    @param_length_matcher
    def sin_of_angle_jerk(self, JY, JR):
        """
            @num   : 30
        """
        sinJ  = self._divisor(JY, JR)
        return sinJ

    @param_length_matcher
    def cos_of_angle_jerk(self, JX, JR):
        """
            @num   : 31
        """
        cosJ = self._divisor(JX, JR)
        return cosJ

    @param_length_matcher
    def angle_of_centripetal_jerk(self, JX, JY):
        """
            @num   : 32
        """
        pass

    @param_length_matcher
    def sin_of_angle_jerk_c(self, JC, JR):
        """
            @num   : 33
        """
        pass

    @param_length_matcher
    def cos_of_angle_jerk_t(self, JT, JR):
        """
            @num   : 34
        """
        pass

    @param_length_matcher
    def velocity_of_pressure(self, P, T):
        """
            @num   : 36
        """
        VP = self._derivative(P, T)
        return VP

    @param_length_matcher
    def acc_of_pressure(self, VP, T):
        """
            @num   : 37
        """
        AP = self._derivative(VP, T)
        return AP

    @param_length_matcher
    def curvature(self, VX, AY, VY, AX, VR):
        """
            @num   : 44
        """
        tempList = [VX[i]*AX[i]*VY[i]*AY[i]/VR[i]/pow(VR[i], 3) for i in range(len(VX))]
        return numpy.log(tempList)

if __name__ == "__main__":
    processor = PreProcessor()
    X = [[1, 1.2, 1.3, 1.1]]
