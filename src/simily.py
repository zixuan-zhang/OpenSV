#coding: utf-8

import os
import numpy

from sklearn import svm

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
        refDis = []

        for i in range(8):
            Xi = self.reference[i][0]
            Yi = self.reference[i][1]
            dis = 0.0
            for j in range(8):
                if i == j:
                    continue
                Xj = self.reference[j][0]
                Yj = self.reference[j][1]
                dis = self.similarity.calculate_2D(Xi, Yi, Xj, Yj)
            refDis.append(dis)

        templateIndex = refDis.insert(min(refDis))
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
                Xj = self.reference[j][0]
                Yj = self.reference[j][1]
                dis = self.similarity.calculate_2D(Xi, Yi, Xj, Yj)
                disList.append(disList)
            minDisList.append(min(disList))
            maxDisList.append(max(disList))

        self.tempDis = numpy.mean(tempDisList)
        self.minDis = numpy.mean(minDisList)
        self.maxDis = numpy.mean(maxDisList)

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

def Driver():

    def __init__(self):
        signatures = self.get_data()
        signatures = self.calculate_delta(signatures)
        train_set, test_set = self.train_test_split()

        self.svm = svn.SVC()

        genuineX = []
        forgeryX = []

        genuineY = []
        forgeryY = []

        # 现在就可以进入训练阶段了
        for sigs in train_set:
            personTrain = PersonTraining(sigs)
            geuinue, forgery = personTrain.calc_train_set())
            genuineX.extend(genuine)
            forgeryX.extend(forgery)

        genuineY = [1] * len(genuineX)
        forgeryY = [0] * len(forgeryX)
        trainX = genuineX + forgeryX
        trainY = genuineY = forgeryY

        self.svm.fit(trainX, trainY)

    def predict(self, X):
        self.svm.predict(X)

    def get_data():
        """
        获取签名样本
        """
        signatures = []
        dataPath = "../data/data"
        for uid in range(1, 41):
            personSigs = []
            for sig in range(1, 41):
                fileName = "U%dS%d.TXT"
                filePath = os.path.join(dataPath, fileName)
                X, Y, T, P = utils.get_data_from_file(filePath)
                personSig.append([X, Y])
            signatures.append(personSigs)
        return signatures

    def calculate_delta(signatures):
        """
        将原始的x, y坐标变换为deltaX, deltaY
        """
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

    def train_test_split(signatures):
        """
        return training_set & test_set
        """
        return signatures[0:20], signatures[21, 40]

def test_DTW():
    X = [3, 5, 6, 7, 7, 1]
    Y = [3, 6, 6, 7, 8, 1, 1]
    Z = [2, 5, 7, 7, 7, 7, 2]

    dtw = DynamicTimeWrappingSimilarity(0,1)
    dtw.calculate(X, Y)
    dtw.calculate(X, Z)

if __name__ == "__main__":
   test_DTW() 
