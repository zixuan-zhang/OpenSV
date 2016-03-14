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


class DataProcessor(object):

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
        D.append(D[-1]) #为了保证D的结果与A和T长度一致
        return D

    def _divisor(self, A, B):
        """
            除法子函数
        """
        D = [A[i] / B[i] for i in range(len(A))]
        return D


class DCTProcessor(DataProcessor):

    """
    @Des: This class implemented some signals generated from X, Y, P

    @Introduction:
        X for x axis
        Y for y axis
        V for velocity
        A for absolute or acceleration
        tan for angle
        sin for sinuous
        cos for cosine
        T for Tangential
        C for centripetal
        J for jerk


    @Detail Table:
    1. X    : coordinate of x axis
    2. Y    : coordinate of y axis
    3. R    : absolute position r
    4. VX   : velocity in x axis
    5. VY   : velocity in y axis
    6. AV   : absolute velocity
    7. VR   : velocity of absolute position r
    8. tanV : angle of velocity
    9. sinV : sinuous of angle velocity
    10.cosV : cosine of angle velocity
    11.AX   : acceleration of x axis
    12.AY   : acceleration of y axis
    13.AA   : absolute acceleration
    14.TA   : tangential acceleration
    15.CA   : centripetal acceleration
    16.AR   : acceleration of r
    17.tanA : angle of acceleration
    18.sinA : sinuous of angle acceleration
    19.cosA : cosine of angle acceleration
    20.tanCA: angle of centripetal acceleration
    21.
    22.
    23.JX   : jerk in x
    24.JY   : jerk in y
    25.AJ   : absolute jerk
    26.TJ   : tangential jerk
    27.CJ   : centripetal jerk
    28.JR   : jerk in r
    29.tanJ : angle of jerk
    30.sinJ : sinuous of angle jerk
    31.cosJ : cosine of angle jerk
    32.tanCJ: angle of centripetal jerk
    33. 
    34.
    35.P    : pressure
    36.VP   : velocity of pressure
    37.AP   : acceleration of pressure
    38.
    39.
    40.
    41.
    42.
    43.
    44.Cur: curvature
    """

    @param_length_matcher
    def radius(self, X, Y):
        """
            @num   : 3
            @des   : calculate radius signal
            @input : X for x axis # list
                     Y for y axis # list
            @output: R for radius # list
        """
        R = self._sqrt(X, Y)
        return R

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
        return VR

    @param_length_matcher
    def angle_of_velocity(self, VX, VY):
        """
            @num   : 8
        """
        tanV = self._divisor(VY, VX)
        return tanV

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
        AX = self._derivative(VX, T)
        return AX

    @param_length_matcher
    def acc_of_vy(self, VY, T):
        """
            @num   : 12
        """
        AY = self._derivative(VY, T)
        return AY
        
    @param_length_matcher
    def abs_acc(self, AX, AY):
        """
            @num   : 13
        """
        AA = self._sqrt(AX, AY)
        return AA

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
        return CA

    @param_length_matcher
    def acc_of_radius(self, VR, T):
        """
            @num   : 16
        """
        AR = self._derivative(VR, T)
        return AR

    @param_length_matcher
    def angle_of_acc(self, AX, AY):
        """
            @num   : 17
        """
        tanA = self._divisor(AY, AX)
        return tanA

    @param_length_matcher
    def sin_of_angle_a(self, AY, AR):
        """
            @num   : 18
        """
        sinA = self._derivative(AY, AR)
        return sinA

    @param_length_matcher
    def cos_of_angle_a(self, AX, AR):
        """
            @num   : 19
        """
        cosA = self._derivative(AX, AR)
        return cosA

    @param_length_matcher
    def angle_of_centripetal_acc(self, AX, AY):
        """
            @num   : 20
        """
        tanCA = self._divisor(AY, AX)
        return tanCA
        
    
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
        AJ = self._sqrt(JX, JY)
        return AJ

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
        tanJ = self._divisor(JY, JX)

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
        Cur = numpy.log(tempList)
        return Cur

class SVMProcessor(DataProcessor):

    def _abs(self, ListV):
        ListAbsV = [[abs(v) for v in V] for V in ListV]
        return ListAbsV

    def _velocity_of_list(self, ListA, ListT):
        ListV = []
        for i in range(len(ListA)):
            V = self._derivative(ListA[i], ListT[i])
            ListV.append(V)
        return ListV

    def _acc_of_v(self, ListV, ListT):
        ListA = []
        for i in range(len(ListV)):
            A = self._derivative(ListV[i], ListT[i])
            ListA.append(A)
        return ListA

    @param_length_matcher
    def radius(self, ListX, ListY):
        """
            @num   : 3
            @output: R for radius # list
        """
        ListR = []
        for i in range(len(ListX)):
            R = []
            for j in range(len(ListX[i]) - 1):
                R.append(numpy.sqrt(pow(ListX[i][j+1]-ListX[i][j], 2) + 
                    pow(ListY[i][j+1]-ListY[i][j], 2)))
            R.append(R[-1])
            ListR.append(R)
        return ListR

    @param_length_matcher
    def velocity_of_x(self, ListX, ListT):
        """
            @num   : 4
            @output: V for velocity
        """
        ListVX = self._velocity_of_list(ListX, ListT)
        return ListVX

    @param_length_matcher
    def velocity_of_y(self, ListY, ListT):
        """
            @num   : 5
        """
        ListVY = self._velocity_of_list(ListY, ListT)
        return ListVY

    @param_length_matcher
    def velocity_of_r(self, ListR, ListT):
        """
            @num   : 6
        """
        # TODO: equation error
        #ListVR = self._velocity_of_list(ListR, ListT)
        ListVR = []
        for i in range(len(ListR)):
            R = [ListR[i][j] / (ListT[i][j+1]-ListT[i][j]) for j in range(len(ListR[i]) - 1)]
            R.append(R[-1])
            ListVR.append(R)
        return ListVR

    @param_length_matcher
    def abs_velocity_of_x(self, ListVX):
        """
            @num   : 7
        """
        ListAbsVX = self._abs(ListVX)
        return ListAbsVX

    @param_length_matcher
    def abs_velocity_of_y(self, ListVY):
        """
            @num   : 8
        """
        ListAbsVY = self._abs(ListVY)
        return ListAbsVY

    @param_length_matcher
    def abs_velocity_of_r(self, ListVR):
        """
            @num   : 9
        """
        ListAbsVR = self._abs(ListVR)
        return ListAbsVR

    @param_length_matcher
    def acc_of_vx(self, ListVX, ListT):
        """
            @num   : 10 
        """
        ListAX = self._acc_of_v(ListVX, ListT)
        return ListAX

    @param_length_matcher
    def acc_of_vy(self, ListVY, ListT):
        """
            @num   : 11
        """
        ListAY = self._acc_of_v(ListVY, ListT)
        return ListAY

    @param_length_matcher
    def acc_of_vr(self, ListVR, ListT):
        """
            @num   : 12
        """
        ListAR = self._acc_of_v(ListVR, ListT)
        return ListAR

    @param_length_matcher
    def abs_acc_of_x(self, ListAX):
        """
            @num   : 13
        """
        ListAbsAX = self._abs(ListAX)
        return ListAbsAX

    def abs_acc_of_y(self, ListAY):
        """
            @num   : 14
        """
        ListAbsAY = self._abs(ListAY)
        return ListAbsAY

    def abs_acc_of_r(self, ListAR):
        """
            @num   : 15
        """
        ListAbsAR = self._abs(ListAR)
        return ListAbsAR

    def velocity_of_p(self, ListP, ListT):
        """
            @num   : 17
        """
        ListVP = self._velocity_of_list(ListP, ListT)
        return ListVP

    def abs_velocity_of_p(self, ListVP):
        """
            @num   : 18
        """
        ListAbsVP = self._abs(ListVP)
        return ListAbsVP

    @param_length_matcher
    def abs_velocity(self, ListVX, ListVY):
        """
            @num   : 6
            @des   : calculate absolute velocity
            @input : VX for x axis velocity 
                   : VY for y axis velocity
            @output: AV for absolute velocity
        """
        ListAV = []
        for i in range(len(ListVX)):
            AV = self._sqrt(ListVX[i], ListVY[i])
            ListAV.append(AV)
        return ListAV

class VagueProcessor(object):

    def __init__(self):
        pass

    def vague_processor(self, X, Y):
        """
            需要子类实现vague方法
        """

    def do_vague(self, X, Y):
        RX, RY = self.vague_processor(X, Y)
        return RX, RY

class RawVagueProcessor(VagueProcessor):

    def __init__(self, rate = 0.3):
        self.rate = rate

    @param_length_matcher
    def vague_processor(X, Y):
        resultX = []
        resultY = []
        for i in range(len(X)):
            if (i % 10 not in [range(10*rate)]):
                resultX.append(X[i])
                resultY.append(Y[i])
        return resultX, resultY

if __name__ == "__main__":
    processor = PreProcessor()
    X = [[1, 1.2, 1.3, 1.1]]
