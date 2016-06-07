#coding: utf-8

import os
import time
import logging
import numpy
import datetime

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import utils
import processor

FOLDER = "../data"
FILENAME = datetime.datetime.now().strftime("%Y%m%d%H%M%S.log")
FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
logging.basicConfig(filename = "%s/%s" % (FOLDER, FILENAME), level = logging.INFO, format = FORMAT)
LOGGER = logging.getLogger()

"""
Singature Component:
    'X': x axis
    'Y': y axis
    'P': pressure
    'VX': velocity of x axis
    'VY': velocity of y axis
    'AX': acceleration of x axis
    'AY': acceleration of y axis
"""

METHOD = 1
# Signal list which need to be considered
SigCompList = ["Y", "VX", "VY"]
PENALIZATION = {
        "X": 7,
        "Y": 7,
        "VX": 6,
        "VY": 6,
        "P": 7,
        "VP": 2,
        "AX": 7,
        "AY": 7,
        }
THRESHOLD = {
        "X": 2,
        "Y": 2,
        "VX": 2,
        "VY": 0,
        "P": 3,
        "VP": 0,
        "AX": 1,
        "AY": 1,
        }
FEATURE_TYPE = {
        "X": ["template", "min", "avg"],
        "Y": ["template", "min", "avg"],
        "VX": ["template","min", "avg"],
        "VY": ["template","min", "avg"],
        "P": ["template", "min", "avg"],
        "VP": ["template", "min", "avg"],
        "AX": ["template", "min", "avg"],
        "AY": ["template", "min", "avg"],
        }
PERSON_COUNT = 40
SIG_COUNT = 40
GENUINE_COUNT = 20
TRAINING_SET_COUNT = 20
REF_COUNT = 8
CLASSIFIER = "RFC" # "RFC", "GBC", "SVM", "MLP"

# Random Forest Tree settings
MAX_DEPTH = 3
MAX_FEATURES = None
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 1
N_JOBS = 1

LOCAL_NORMAL_TYPE = "mid" # "mid" or "offset"
RANDOM_FORGERY_INCLUDE = False
TRAIN_SET_INCLUDE = False
SIZE_NORM_SWITCH = True

LOGGER.info("SizeNormalizationSwitch: %s" % SIZE_NORM_SWITCH)
LOGGER.info("RandomForgeryInclude: %s" % RANDOM_FORGERY_INCLUDE)
LOGGER.info("ClassifierType: %s" % CLASSIFIER)
LOGGER.info("LocalNormalizationType: %s" % LOCAL_NORMAL_TYPE)
LOGGER.info("TrainingSetCount: %d" % TRAINING_SET_COUNT)
LOGGER.info("Reference Count: %d" % REF_COUNT)
LOGGER.info("Method: %d" % METHOD)
LOGGER.info("Signal List: %s" % SigCompList)
LOGGER.info("PENALIZATION: %s" % PENALIZATION)
LOGGER.info("THRESHOLD: %s" % THRESHOLD)
LOGGER.info("FEATURE_TYPE: %s" % FEATURE_TYPE)
LOGGER.info("RandomForestTree: max_feature: %s, n_estimator: %d, min_sample_leaf: %d, n_jobs: %d, max_depth: %d" %
        (MAX_FEATURES, N_ESTIMATORS, MIN_SAMPLES_LEAF, N_JOBS, MAX_DEPTH))

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
            elif METHOD == 3:
                d1 = distance[i-1][j] + abs(A[i] - B[j])
                d2 = distance[i][j-1] + abs(A[i] - B[j])
                other = 0 if (abs(A[i] - B[j]) < threshold) else (abs(A[i] - B[j]) - threshold)
                d3 = distance[i-1][j-1] + other
                distance[i][j] = min([d1, d2, d3])
                
    return distance[len1-1][len2-1]

class Person(object):

    def __init__(self, refSigs, testSigs, key = None):

        self.key = key
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
            avgComList = []
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
                avgComList.append(numpy.mean(comDisList))
            if "template" in FEATURE_TYPE[com]:
                self.base["template" + com] = numpy.mean(templateComList)
            if "max" in FEATURE_TYPE[com]:
                self.base["max"+com] = numpy.mean(maxComList)
            if "min" in FEATURE_TYPE[com]:
                self.base["min"+com] = numpy.mean(minComList)
            if "avg" in FEATURE_TYPE[com]:
                self.base["avg"+com] = numpy.mean(avgComList)
            LOGGER.info("Calculating signal: %s. %s" % (com, ", ".join(["%s:%s"%(items[0], items[1]) for items in self.base.items()])))

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
            avgComDis = numpy.mean(comDisList)
            if "template" in FEATURE_TYPE[com]:
                featureVec.append(templateComDis / self.base["template"+com])
            if "max" in FEATURE_TYPE[com]:
                featureVec.append(maxComDis / self.base["max"+com])
            if "min" in FEATURE_TYPE[com]:
                featureVec.append(minComDis / self.base["min"+com])
            if "avg" in FEATURE_TYPE[com]:
                featureVec.append(avgComDis / self.base["avg"+com])
        return featureVec

class PersonTest(Person):
    def __init__(self, refSigs):
        super(PersonTest, self).__init__(refSigs, None)

class PersonTraining(Person):
    def __init__(self, signatures):
        """
        """
        # eight reference signatures
        super(PersonTraining, self).__init__(signatures[0:REF_COUNT], signatures[REF_COUNT:])
        
        self.genuineSigs = self.testSigs[:GENUINE_COUNT-REF_COUNT]
        self.forgerySigs = self.testSigs[GENUINE_COUNT-REF_COUNT:]

        LOGGER.info("Reference signature count: %d, test signature count: %d, \
                genuine test signatures: %d, forgery test signatures: %d" % \
                (self.refCount, len(self.testSigs), len(self.genuineSigs), len(self.forgerySigs)))

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
        signatures = self.get_data_from_task2()
        signatures = self.pre_process(signatures)
        LOGGER.info("Total signatures: %d" % len(signatures))
        signatures = self.reconstructSignatures(signatures)
        self.train_set, self.test_set = self.train_test_split(signatures)
        LOGGER.info("Spliting value set, training_set %d, test_set %d" % (len(self.train_set), len(self.test_set)))

        if CLASSIFIER == "SVM":
            self.svm = svm.SVC()
        elif CLASSIFIER == "GBC":
            self.svm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
        elif CLASSIFIER == "RFC":
            # self.svm = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS,
                # max_features = MAX_FEATURES, min_samples_leaf = MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH)
            self.svm = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS)

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
        for uid in range(PERSON_COUNT):
            uSigs = []
            for sid in range(SIG_COUNT):
                RX = signatures[uid][sid][0]
                RY = signatures[uid][sid][1]
                P = signatures[uid][sid][2]
                if SIZE_NORM_SWITCH:
                    RX, RY = self.processor.size_normalization(RX, RY, 400, 200)
                if LOCAL_NORMAL_TYPE == "mid":
                    RX, RY = self.processor.location_normalization(RX, RY)
                elif LOCAL_NORMAL_TYPE == "offset":
                    RX, RY = self.processor.offset_to_origin_normalization(RX, RY)
                uSigs.append([RX, RY, P])
            result.append(uSigs)
        return result

    def get_data_from_task2(self):
        """
        Load original data from svc2004 task2
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
                personSigs.append([X, Y, P])
            signatures.append(personSigs)
        return signatures

    def get_data_from_susig(self):
        """
        Load original data from susig
        """

    def reconstructSignatures(self, signatures):
        """
        Reconstruct signatures to dictionary like object.
        """
        reconstructedSigs = []
        for uid in range(PERSON_COUNT):
            uSigs = []
            for sid in range(SIG_COUNT):
                signature = signatures[uid][sid]
                X = signature[0]
                Y = signature[1]
                P = signature[2]
                VX = self.calculate_delta(X)
                VY = self.calculate_delta(Y)
                VP = self.calculate_delta(P)
                AX = self.calculate_delta(VX)
                AY = self.calculate_delta(VY)
                uSigs.append({"X": X, "Y": Y, "P": P, "VX": VX, "VY": VY, "VP": VP, "AX": AX, "AY": AY})
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
        return signatures[0:trainCount], signatures[trainCount:PERSON_COUNT]

    def test(self):
        LOGGER.info("Start test")
        count = 1
        test_set = self.test_set
        if TRAIN_SET_INCLUDE:
            test_set.extend(self.train_set)
        forgery_test_result = []
        genuine_test_result = []
        random_test_result = []
        for i in range(len(test_set)):
            one_test_set = test_set[i]
            LOGGER.info("Test signature: %d" % count)
            count += 1
            personTest = PersonTest(one_test_set[0:REF_COUNT])
            genuine_set = one_test_set[REF_COUNT:GENUINE_COUNT]
            forgery_set = one_test_set[GENUINE_COUNT:PERSON_COUNT]
            random_set = []

            for j in range(len(genuine_set)):
                sig = genuine_set[j]
                dis = personTest.calc_dis(sig)
                res = self.svm.predict(dis)
                LOGGER.info("Genuine Test: Result: %s, %s" % (res, dis))
                genuine_test_result.append(res)
                if (res != 1):
                    LOGGER.fatal("FalseReject: uid: %d, sid: %d" % (i, j))

            for j in range(len(forgery_set)):
                sig = forgery_set[j]
                dis = personTest.calc_dis(sig)
                res = self.svm.predict(dis)
                LOGGER.info("Forgery Test: Result: %s, %s" % (dis, res))
                forgery_test_result.append(res)
                if (res != 0):
                    LOGGER.fatal("FalseAccept: uid: %d, sid: %d" % (i, j))

            if RANDOM_FORGERY_INCLUDE:
                for j in range(len(test_set)):
                    if i == j:
                        continue
                    random_set.extend(test_set[j])
                # train set included
                for one_train_set in self.train_set:
                    random_set.extend(one_train_set)

                for j in range(len(random_set)):
                    sig = random_set[j]
                    dis = personTest.calc_dis(sig)
                    res = self.svm.predict(dis)
                    LOGGER.info("Random Test: Result: %s, %s" % (res, dis))
                    random_test_result.append(res)
                    if (res != 0):
                        LOGGER.fatal("FalseAccept: uid: %d, sig: %d" % (i, j))

        LOGGER.info("genuine test set count: %d" % len(genuine_test_result))
        LOGGER.info("true accepted count: %d" % sum(genuine_test_result))
        LOGGER.info("false rejected rate: %f" % (sum(genuine_test_result) / float(len(genuine_test_result))))

        LOGGER.info("forgery test set count: %d" % len(forgery_test_result))
        LOGGER.info("false accepted count: %d" % sum(forgery_test_result))
        LOGGER.info("false accepted rate: %f" % (1 - sum(forgery_test_result) / float(len(forgery_test_result))))

        if RANDOM_FORGERY_INCLUDE:
            LOGGER.info("random test set count: %d" % len(random_test_result))
            LOGGER.info("false accepted count: %d" % sum(random_test_result))
            LOGGER.info("false accepted rate: %f" % (1 - sum(random_test_result) / float(len(random_test_result))))

def test_DTW():
    start = time.time()
    driver = Driver()
    driver.test()
    end = time.time()
    LOGGER.info("Total time : %f" % (end - start))

if __name__ == "__main__":
   test_DTW() 
