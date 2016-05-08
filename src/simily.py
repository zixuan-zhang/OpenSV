#coding: utf-8

import os
import logging
import numpy

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import utils
import processor

FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
logging.basicConfig(filename = '../data/log', level = logging.INFO, format = FORMAT)
LOGGER = logging.getLogger()

"""
Singature Component:
    'X': x axis
    'Y': y axis
    'VX': velocity of x axis
    'VY': velocity of y axis
"""

SigCompList = ["X", "Y", "VX", "VY"]

def naive_dtw(A, B, p=5, t=5):
    penalization = p
    threshold = t
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
            # method 1
            distance[i][j] = min([distance[i-1][j], distance[i][j-1],
                    distance[i-1][j-1]]) + abs(A[i]-B[j])

            """
            # method 2
            d1 = distance[i-1][j] + penalization
            d2 = distance[i][j-1] + penalization
            other = 0 if (abs(A[i] - B[j]) < theshold) else (abs(A[i] - B[j]) - theshold)
            d3 = distance[i-1][j-1] + other
            distance[i][j] = min([d1, d2, d3])
            """

    return distance[len1-1][len2-1]

class Person(object):

    def __init__(self, refSigs, testSigs):

        self.refSigs = refSigs
        self.testSigs = testSigs
        self.refCount = len(refSigs)

        self.templateSig = None

        # select template signature
        self.select_template()

        # calculate base distance
        self.calc_base_dis()

    def select_template(self):
        """
        Just select template signature.
        TODO: different signal with variant weight
        """
        LOGGER.info("selecting template signature")
        refDis = []
        for i in range(self.refCount):
            dis = 0.0
            for j in range(self.refCount):
                if i == j:
                    continue
                comDisList = []
                for com in SigCompList:
                    signal1 = self.refSigs[i][com]
                    signal2 = self.refSigs[j][com]
                    comDisList.append(naive_dtw(signal1, signal2))
                dis += numpy.mean(comDisList)
            refDis.append(dis)
        templateIndex = refDis.index(min(refDis))
        LOGGER.info("template index : %d" % templateIndex)
        self.templateSig = self.refSigs.pop(templateIndex)
        self.refCount -= 1

    def calc_base_dis(self):
        """
        Calculate the base value in signal component of signatures
        """
        LOGGER.info("Calculating base distance")
        self.base = {}

        for com in SigCompList:
            signalBaseList = []


        for i in range(self.refCount):
            for com in SigCompList:
                signal1 = self.templateSig[com]
                signal2 = self.refCount[i][com]

        for i in range(self.refCount):
            Xi = self.refSigs[i][0]
            Yi = self.refSigs[i][1]
            templateXDis = naive_dtw(Xi, templateX)
            templateYDis = naive_dtw(Yi, templateY)
            templateXList.append(templateXDis)
            templateYList.append(templateYDis)

            xDisList = []
            yDisList = []
            for j in range(self.refCount):
                if i == j:
                    continue
                Xj = self.refSigs[j][0]
                Yj = self.refSigs[j][1]
                xDis = naive_dtw(Xi, Xj)
                yDis = naive_dtw(Yi, Yj)
                xDisList.append(xDis)
                yDisList.append(yDis)
            minXList.append(min(xDisList))
            minYList.append(min(yDisList))
            maxXList.append(max(xDisList))
            maxYList.append(max(yDisList))

        self.baseTemplateXDis = numpy.mean(templateXList)
        self.baseTemplateYDis = numpy.mean(templateYList)
        self.baseMinXDis = numpy.mean(minXList)
        self.baseMinYDis = numpy.mean(minYList)
        self.baseMaxXDis = numpy.mean(maxXList)
        self.baseMaxYDis = numpy.mean(maxYList)

        LOGGER.info("Calculation done. baseTemplateX: %f, baseTemplateY: %f, baseMinX: %f, baseMinY: %f, baseMaxX: %f, baseMaxY: %f" % (self.baseTemplateXDis, self.baseTemplateYDis, self.baseMinXDis, self.baseMinYDis, self.baseMaxXDis, self.baseMaxYDis))


    def calc_dis(self, X, Y):
        """
        For given signature, calculate vector[tempX, tempY, minX, minY, maxX, maxY] with normalization
        """
        templateX = self.templateSig[0]
        templateY = self.templateSig[1]

        templateXDis = naive_dtw(X, templateX)
        templateYDis = naive_dtw(Y, templateY)

        xDisList = []
        yDisList = []
        for sig in self.refSigs:
            RX = sig[0]
            RY = sig[1]
            xDis = naive_dtw(X, RX)
            yDis = naive_dtw(Y, RY)
            xDisList.append(xDis)
            yDisList.append(yDis)
        minXDis = min(xDisList)
        minYDis = min(yDisList)
        maxXDis = max(xDisList)
        maxYDis = max(yDisList)

        return [templateXDis / self.baseTemplateXDis, templateYDis / self.baseTemplateYDis, minXDis / self.baseMinXDis, minYDis / self.baseMinYDis, maxXDis / self.baseMaxXDis, maxYDis / self.baseMaxYDis]


class PersonTest(Person):
    def __init__(self, refSigs):
        super(PersonTest, self).__init__(refSigs, None)


class PersonTraining(Person):
    def __init__(self, signatures):
        """
        suppose 40 signatures: 20 genuine & 20 forgeries
        """
        # eight reference signatures
        super(PersonTraining, self).__init__(signatures[0:8], signatures[8:])
        
        self.genuineSigs = self.testSigs[:12]
        self.forgerySigs = self.testSigs[12:]

        LOGGER.info("Reference signature count: %d, test signature count: %d, genuine test signatures: %d, forgery test signatures: %d" % (self.refCount, len(self.testSigs), len(self.genuineSigs), len(self.forgerySigs)))

    def calc_train_set(self):
        """
        For given training sets, calculate True samples & False samples
        """
        LOGGER.info("Calculating training vectors")
        genuineVec = []
        forgeryVec = []

        for genuine in self.genuineSigs:
            X = genuine[0]
            Y = genuine[1]
            genuineVec.append(self.calc_dis(X, Y))

        for forgery in self.forgerySigs:
            X = forgery[0]
            Y = forgery[1]
            forgeryVec.append(self.calc_dis(X, Y))

        return genuineVec, forgeryVec

class Driver():
    def __init__(self):

        signatures = self.get_data()
        signatures = self.pre_process(signatures)
        LOGGER.info("Total signatures: %d" % len(signatures))
        signatures = self.reconstructSignatures(signatures)
        self.train_set, self.test_set = self.train_test_split(signatures)
        LOGGER.info("Spliting value set, training_set %d, test_set %d" % (len(self.train_set), len(self.test_set)))

        # self.svm = svm.SVC()
        # self.svm = tree.DecisionTreeClassifier()
        self.svm = RandomForestClassifier(n_estimators=50)

        genuineX = []
        forgeryX = []

        genuineY = []
        forgeryY = []

        # Training process
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

    def pre_process(self, signatures):
        """
        apply size normalization and localtion normalization
        """
        self.processor = processor.PreProcessor()
        result = []
        for uid in range(40):
            uSigs = []
            for sid in range(40):
                X = signatures[uid][sid][0]
                Y = signatures[uid][sid][1]
                RX, RY = self.processor.size_normalization(X, Y, 400, 200)
                RX, RY = self.processor.location_normalization(RX, RY)
                uSigs.append([RX, RY])
            result.append(uSigs)
        return result

    def get_data(self):
        """
        获取签名样本
        """
        LOGGER.info("Getting signatures")
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

    def reconstructSignatures(self, signatures):
        """
        Reconstruct signatures to dictionary like object.
        """
        reconstructedSigs = []
        for signature in signatures:
            X = signature[0]
            Y = signature[1]
            VX = calculate_delta(X)
            VY = calculate_delta(Y)
            reconstructedSig = {"X": X, "Y": Y, "VX": VX, "VY": VY}
            reconstructedSigs.append(reconstructedSig)
        return reconstructedSigs

    def calculate_delta(self, valueList):
        deltaValueList = [valueList[i] - valueList[i-1] for i in range(1, len(valueList))]
        return deltaValueList

    def train_test_split(self, signatures):
        """
        return training_set & test_set
        """
        trainCount = 20
        return signatures[0:trainCount], signatures[trainCount:40]

    def test(self):
        test_set = self.test_set
        forgery_test_result = []
        genuine_test_result = []
        for one_test_set in test_set:
            personTest = PersonTest(one_test_set[0:8])
            genuine_set = one_test_set[8:20]
            forgery_set = one_test_set[20:40]

            for sig in genuine_set:
                X = sig[0]
                Y = sig[1]
                dis = personTest.calc_dis(X, Y)
                genuine_test_result.append(self.svm.predict(dis))

            for sig in forgery_set:
                X = sig[0]
                Y = sig[1]
                dis = personTest.calc_dis(X, Y)
                forgery_test_result.append(self.svm.predict(dis))

        LOGGER.info("genuine test set count: %d" % len(genuine_test_result))
        LOGGER.info("true accepted count: %d" % sum(genuine_test_result))
        LOGGER.info("false rejected rate: %f" % (sum(genuine_test_result) / float(len(genuine_test_result))))

        LOGGER.info("forgery test set count: %d" % len(forgery_test_result))
        LOGGER.info("false accepted count: %d" % sum(forgery_test_result))
        LOGGER.info("false accepted rate: %f" % (1 - sum(forgery_test_result) / float(len(forgery_test_result))))

def test_DTW():
    driver = Driver()
    driver.test()

if __name__ == "__main__":
   test_DTW() 
