#coding: utf-8

import os
import numpy

from sklearn import svm
from sklearn import tree

import utils

class Similarity(object):

    def calculate(self):
        pass

class DynamicTimeWrappingSimilarity(Similarity):

    def naive_dtw(self, A, B):
        """
        朴素的动归算法
        """
        len1 = len(A)
        len2 = len(B)

        distance = numpy.zeros((len1, len2))

        # initialize distance values
        distance[0][0] = abs(A[0] - B[0])
        for i in range(1, len1):
            distance[i][0] = distance[i-1][0] + abs(A[i] - B[0])

        for j in range(1, len2):
            distance[0][j] = distance[0][j-1] + abs(A[0] - B[j])

        for i in range(1, len1):
            for j in range(1, len2):
                distance[i][j] = min([distance[i-1][j], distance[i][j-1],
                        distance[i-1][j-1]]) + abs(A[i]-B[j])

        return distance[len1-1][len2-1]

    def calculate(self, A, B):
        return self.naive_dtw(A, B)

    def calculate_2D(self, Ax, Ay, Bx, By):
        Simx = self.calculate(Ax, Bx)
        Simy = self.calculate(Ay, By)
        return (Simx + Simy) / 2

class PersonTest():
    def __init__(self, refSigs):
        self.refSigs = refSigs
        self.refCount = len(refSigs)
        self.templateSig = None

        self.similarity = DynamicTimeWrappingSimilarity()
        # 选择template signature
        self.select_template()
        # 计算base distance
        self.calc_base_dis()

    def select_template(self):
        print "selecting template signature"

        refDis = []
        for i in range(self.refCount):
            Xi = self.refSigs[i][0]
            Yi = self.refSigs[i][1]
            dis = 0.0
            for j in range(self.refCount):
                if i == j:
                    continue
                Xj = self.refSigs[j][0]
                Yj = self.refSigs[j][1]
                dis += self.similarity.calculate_2D(Xi, Yi, Xj, Yj)
            refDis.append(dis)

        templateIndex = refDis.index(min(refDis))
        print "template index : %d" % templateIndex
        self.templateSig = self.refSigs.pop(templateIndex)
        self.refCount -= 1

    def calc_base_dis(self):
        """
        计算用于归一化的templateDis, minDis, maxDis
        """
        tempDisList = []
        minDisList = []
        maxDisList = []
        tempX = self.templateSig[0]
        tempY = self.templateSig[1]
        for i in range(self.refCount):
            Xi = self.refSigs[i][0]
            Yi = self.refSigs[i][1]
            tempDis = self.similarity.calculate_2D(Xi, Yi, tempX, tempY)
            tempDisList.append(tempDis)

            disList = []
            for j in range(self.refCount):
                if i == j:
                    continue
                Xj = self.refSigs[j][0]
                Yj = self.refSigs[j][1]
                dis = self.similarity.calculate_2D(Xi, Yi, Xj, Yj)
                disList.append(dis)
            minDisList.append(min(disList))
            maxDisList.append(max(disList))

        self.tempDis = numpy.mean(tempDisList)
        self.minDis = numpy.mean(minDisList)
        self.maxDis = numpy.mean(maxDisList)
        print "templateDis %f, minDis %f, maxDis %f" % (self.tempDis,
                self.minDis, self.maxDis)

    def calc_dis(self, X, Y):
        """
        对于任意signature样本，计算template distance, min distance, max distance
        的归一化后的结果
        """
        tempX = self.templateSig[0]
        tempY = self.templateSig[1]
        tempDis = self.similarity.calculate_2D(X, Y, tempX, tempY)

        disList = []
        for sig in self.refSigs:
            Rx = sig[0]
            Ry = sig[1]
            dis = self.similarity.calculate_2D(X, Y, Rx, Ry)
            disList.append(dis)
        maxDis = max(disList)
        minDis = min(disList)
        return [maxDis/self.maxDis, minDis/self.minDis, tempDis/self.tempDis]

class PersonTraining():
    def __init__(self, signatures):
        """
        suppose 40 signatures: 20 genuine & 20 forgeries
        """
        # eight reference signatures
        self.reference = signatures[0:8]
        self.refCount = 8
        self.templateSig = None

        self.genuineSigs = signatures[8:20]
        self.forgerySigs = signatures[20:40]

        self.similarity = DynamicTimeWrappingSimilarity()

        # 选择template signature
        self.select_template()
        # 计算base distance
        self.calc_base_dis()

    def select_template(self):
        print "selecting template signature"

        refDis = []
        for i in range(self.refCount):
            Xi = self.reference[i][0]
            Yi = self.reference[i][1]
            dis = 0.0
            for j in range(self.refCount):
                if i == j:
                    continue
                Xj = self.reference[j][0]
                Yj = self.reference[j][1]
                dis += self.similarity.calculate_2D(Xi, Yi, Xj, Yj)
            refDis.append(dis)

        templateIndex = refDis.index(min(refDis))
        print "template index : %d" % templateIndex
        self.templateSig = self.reference.pop(templateIndex)
        self.refCount -= 1

    def calc_base_dis(self):
        """
        计算用于归一化的templateDis, minDis, maxDis
        """
        tempDisList = []
        minDisList = []
        maxDisList = []
        tempX = self.templateSig[0]
        tempY = self.templateSig[1]
        for i in range(self.refCount):
            Xi = self.reference[i][0]
            Yi = self.reference[i][1]
            tempDis = self.similarity.calculate_2D(Xi, Yi, tempX, tempY)
            tempDisList.append(tempDis)

            disList = []
            for j in range(self.refCount):
                if i == j:
                    continue
                Xj = self.reference[j][0]
                Yj = self.reference[j][1]
                dis = self.similarity.calculate_2D(Xi, Yi, Xj, Yj)
                disList.append(dis)
            minDisList.append(min(disList))
            maxDisList.append(max(disList))

        self.tempDis = numpy.mean(tempDisList)
        self.minDis = numpy.mean(minDisList)
        self.maxDis = numpy.mean(maxDisList)
        print "templateDis %f, minDis %f, maxDis %f" % (self.tempDis,
                self.minDis, self.maxDis)

    def calc_dis(self, X, Y):
        """
        对于任意signature样本，计算template distance, min distance, max distance
        的归一化后的结果
        """
        tempX = self.templateSig[0]
        tempY = self.templateSig[1]
        tempDis = self.similarity.calculate_2D(X, Y, tempX, tempY)

        disList = []
        for sig in self.reference:
            Rx = sig[0]
            Ry = sig[1]
            dis = self.similarity.calculate_2D(X, Y, Rx, Ry)
            disList.append(dis)
        maxDis = max(disList)
        minDis = min(disList)

        return [maxDis/self.maxDis, minDis/self.minDis, tempDis/self.tempDis]

    def calc_train_set(self):
        """
        对于每个训练个体，计算该个体的正确的样本和伪造的样本集
        """
        genuineDis = []
        forgeryDis = []

        for genuine in self.genuineSigs:
            Sx = genuine[0]
            Sy = genuine[1]
            genuineDis.append(self.calc_dis(Sx, Sy))
        for forgery in self.forgerySigs:
            Sx = forgery[0]
            Sy = forgery[1]
            forgeryDis.append(self.calc_dis(Sx, Sy))

        return genuineDis, forgeryDis

class Driver():
    def __init__(self):
        signatures = self.get_data()
        print "Total signatures: %d" % len(signatures)
        signatures = self.calculate_delta(signatures)
        print "Calculating delta done"
        self.train_set, self.test_set = self.train_test_split(signatures)
        print "Spliting value set, training_set %d, test_set %d" % (len(self.train_set), len(self.test_set))

        # self.svm = svm.SVC()
        self.svm = tree.DecisionTreeClassifier()

        genuineX = []
        forgeryX = []

        genuineY = []
        forgeryY = []

        # 现在就可以进入训练阶段了
        for sigs in self.train_set:
            personTrain = PersonTraining(sigs)
            genuine, forgery = personTrain.calc_train_set()
            genuineX.extend(genuine)
            forgeryX.extend(forgery)

        genuineY = [1] * len(genuineX)
        forgeryY = [0] * len(forgeryX)
        trainX = genuineX + forgeryX
        trainY = genuineY + forgeryY

        self.svm.fit(trainX, trainY)

    def get_data(self):
        """
        获取签名样本
        """
        print "Getting signatures"
        signatures = []
        dataPath = "../data/Task2"
        for uid in range(1, 41):
            personSigs = []
            for sig in range(1, 41):
                fileName = "U%dS%d.TXT" % (uid, sig)
                filePath = os.path.join(dataPath, fileName)
                X, Y, T, P = utils.get_data_from_file(filePath)
                personSigs.append([X, Y])
            signatures.append(personSigs)
        return signatures

    def calculate_delta(self, signatures):
        """
        将原始的x, y坐标变换为deltaX, deltaY
        """
        print "Calculating x axis and y axis delta"
        result = []
        for personSigs in signatures:
            personRes = []
            for sig in personSigs:
                [X,Y] = sig
                deltaX = [X[i]-X[i-1] for i in range(1, len(X))]
                deltaY = [Y[i]-Y[i-1] for i in range(1, len(Y))]
                personRes.append([deltaX, deltaY])
            result.append(personRes)
        return result

    def train_test_split(self, signatures):
        """
        return training_set & test_set
        """
        trainCount = 10
        return signatures[0:trainCount], signatures[trainCount:40]

    def test(self):
        test_set = self.test_set[0:2]
        for one_test_set in test_set:
            personTest = PersonTest(one_test_set[0:8])
            genuine_set = one_test_set[8:20]
            forgery_set = one_test_set[20:40]

            print "genuine sig test"
            for sig in genuine_set:
                X = sig[0]
                Y = sig[1]
                dis = personTest.calc_dis(X, Y)
                print self.svm.predict(dis)

            print "forgery sig test"
            for sig in forgery_set:
                X = sig[0]
                Y = sig[1]
                dis = personTest.calc_dis(X, Y)
                print self.svm.predict(dis)

def test_DTW():
    driver = Driver()
    driver.test()

if __name__ == "__main__":
   test_DTW() 
