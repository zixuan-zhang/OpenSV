#coding: utf-8

import os
import logging
import numpy
import datetime

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import utils
import processor

FILENAME = datetime.datetime.now().strftime("%Y%m%d%H%M.log")
FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
logging.basicConfig(filename = "../data/%s" %FILENAME, level = logging.INFO, format = FORMAT)
LOGGER = logging.getLogger()

"""
Singature Component:
    'X': x axis
    'Y': y axis
    'VX': velocity of x axis
    'VY': velocity of y axis
"""

# Signal list which need to be considered
SigCompList = ["VY"]
#SigCompList = ["X", "Y", "VX", "VY"]
PENALIZATION = {
        "X": 5,
        "Y": 5,
        "VX": 3,
        "VY": 3
        }
THRESHOLD = {
        "X": 2,
        "Y": 2,
        "VX": 1,
        "VY": 1,
        }

METHOD = 2
TRAINING_SET_COUNT = 20
LOGGER.info("TrainingSetCount: %d" % TRAINING_SET_COUNT)
LOGGER.info("Method: %d" % METHOD)
LOGGER.info("Signal List: %s" % SigCompList)
LOGGER.info("PENALIZATION: %s" % PENALIZATION)
LOGGER.info("THRESHOLD: %s" % THRESHOLD)

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
            if METHOD == 1:
                distance[i][j] = min([distance[i-1][j], distance[i][j-1],
                        distance[i-1][j-1]]) + abs(A[i]-B[j])
            elif METHOD == 2:
                # method 2
                d1 = distance[i-1][j] + penalization
                d2 = distance[i][j-1] + penalization
                other = 0 if (abs(A[i] - B[j]) < threshold) else (abs(A[i] - B[j]) - threshold)
                d3 = distance[i-1][j-1] + other
                distance[i][j] = min([d1, d2, d3])
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
                    comDisList.append(naive_dtw(signal1, signal2, PENALIZATION[com], THRESHOLD[com]))
                dis += numpy.mean(comDisList)
            refDis.append(dis)
        templateIndex = refDis.index(min(refDis))
        LOGGER.info("template index : %d. RefSigDisList: %s" % (templateIndex, refDis))
        self.templateSig = self.refSigs.pop(templateIndex)
        self.refCount -= 1

    def calc_base_dis(self):
        """
        Calculate the base value in signal component of signatures
        """
        LOGGER.info("Calculating base distance")
        self.base = {}

        for com in SigCompList:
            templateComList = []
            maxComList = []
            minComList = []
            for i in range(self.refCount):
                comi = self.refSigs[i][com]
                templateComDis = naive_dtw(comi, self.templateSig[com], PENALIZATION[com], THRESHOLD[com])
                templateComList.append(templateComDis)
                comDisList = []
                for j in range(self.refCount):
                    if i == j:
                        continue
                    comj = self.refSigs[j][com]
                    comDisList.append(naive_dtw(comi, comj))
                maxComList.append(max(comDisList))
                minComList.append(min(comDisList))
            self.base["template" + com] = numpy.mean(templateComList)
            self.base["max"+com] = numpy.mean(maxComList)
            self.base["min"+com] = numpy.mean(minComList)
            LOGGER.info("Calculating signal: %s, baseTemplate: %f, baseMax: %f, baseMin: %f" %
                    (com, self.base["template"+com], self.base["max"+com], self.base["min"+com]))

    def calc_dis(self, signature):
        """
        For given signature, calculate vector[] with normalization
        """
        featureVec = []
        for com in SigCompList:
            comSig = signature[com]
            comTem = self.templateSig[com]
            templateComDis = naive_dtw(comSig, comTem, PENALIZATION[com], THRESHOLD[com])
            comDisList = []
            for i in range(self.refCount):
                comI = self.refSigs[i][com]
                dis = naive_dtw(comSig, comI, PENALIZATION[com], THRESHOLD[com])
                comDisList.append(dis)
            maxComDis = max(comDisList)
            minComDis = min(comDisList)
            featureVec.append(templateComDis / self.base["template"+com])
            featureVec.append(maxComDis / self.base["max"+com])
            featureVec.append(minComDis / self.base["min"+com])
        return featureVec

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
            genuineV = self.calc_dis(genuine)
            LOGGER.info("Genuine vector: %s" % genuineV)
            genuineVec.append(genuineV)

        for forgery in self.forgerySigs:
            forgeryV = self.calc_dis(forgery)
            LOGGER.info("Forgery vector: %s" % forgeryV)
            forgeryVec.append(forgeryV)

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
        Apply size normalization and localtion normalization
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
        Load original data from file
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
        for uid in range(40):
            uSigs = []
            for sid in range(40):
                signature = signatures[uid][sid]
                X = signature[0]
                Y = signature[1]
                VX = self.calculate_delta(X)
                VY = self.calculate_delta(Y)
                uSigs.append({"X": X, "Y": Y, "VX": VX, "VY": VY})
            reconstructedSigs.append(uSigs)
        return reconstructedSigs

    def calculate_delta(self, valueList):
        deltaValueList = [valueList[i] - valueList[i-1] for i in range(1, len(valueList))]
        return deltaValueList

    def train_test_split(self, signatures):
        """
        return training_set & test_set
        """
        trainCount = TRAINING_SET_COUNT
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
                dis = personTest.calc_dis(sig)
                res = self.svm.predict(dis)
                LOGGER.info("Genuine Test: %s, Result: %s" % (dis, res))
                genuine_test_result.append(res)

            for sig in forgery_set:
                dis = personTest.calc_dis(sig)
                res = self.svm.predict(dis)
                LOGGER.info("Forgery Test: %s, Result: %s" % (dis, res))
                forgery_test_result.append(res)

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
