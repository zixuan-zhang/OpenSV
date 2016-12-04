#coding: utf-8

import os
import time
import logging
import numpy
import datetime
import traceback

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import utils
import processor

FOLDER = "../data/susig_log"
FILENAME = datetime.datetime.now().strftime("%Y%m%d%H%M%S.log")
FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
logging.basicConfig(filename = "%s/%s" % (FOLDER, FILENAME), level = logging.INFO, format = FORMAT)
LOGGER = logging.getLogger()

LOGGER.info("This one use validation sets, with session2")

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

USE_PCA = False
LinearClassifier = "LogisticRegression"

LOGGER.info("USE_PCA: %s" % USE_PCA)
LOGGER.info("Linear classifier is %s" % LinearClassifier)

# Signal list which need to be considered
SigCompList = ["VX", "VY", "P"]

# Feature settings
FEATURE_TYPE = {
        "VX": ["template", "min", "avg"],
        "VY": ["template", "min", "avg"],
        "P": ["template", "min", "avg"],
        "X": ["template", "med", "min", "avg"],
        "Y": ["template", "med", "min", "avg"],
        "VP": ["template", "min", "avg"],
        "AX": ["template", "min", "avg"],
        "AY": ["template", "min", "avg"],
        }
LOGGER.info("FEATURE_TYPE: %s" % FEATURE_TYPE)

# DTW settings
METHOD = 2
PENALIZATION = {
        "X": 7,
        "Y": 5,
        "VX": 3,
        "VY": 2,
        "P": 2,
        "VP": 2,
        "AX": 3,
        "AY": 3,
        }
THRESHOLD = {
        "X": 8,
        "Y": 6,
        "VX": 0,
        "VY": 1,
        "P": 2,
        "VP": 0,
        "AX": 4,
        "AY": 3,
        }

# Data settings

TRAIN_SET_COUNT = 50
REF_COUNT = 5
CLASSIFIER = "RFC" # "RFC", "GBC", "SVM", "MLP", "Logi"

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
MULTI_SESSION = True
PRE_PROCESSOR_SWITCH = True

LOGGER.info("PreProcessorSwitch: %s" % PRE_PROCESSOR_SWITCH)
LOGGER.info("TrainSetCount: %s" % TRAIN_SET_COUNT)
LOGGER.info("MultiSession: %s" % MULTI_SESSION)
LOGGER.info("SizeNormalizationSwitch: %s" % SIZE_NORM_SWITCH)
LOGGER.info("RandomForgeryInclude: %s" % RANDOM_FORGERY_INCLUDE)
LOGGER.info("ClassifierType: %s" % CLASSIFIER)
LOGGER.info("LocalNormalizationType: %s" % LOCAL_NORMAL_TYPE)
LOGGER.info("Reference Count: %d" % REF_COUNT)
LOGGER.info("Method: %d" % METHOD)
LOGGER.info("Signal List: %s" % SigCompList)
LOGGER.info("PENALIZATION: %s" % PENALIZATION)
LOGGER.info("THRESHOLD: %s" % THRESHOLD)
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

        self.templateIndex = refDis.index(min(refDis))
        LOGGER.info("template index : %d. RefSigDisList: %s" % (self.templateIndex, refDis))
        self.templateSig = self.refSigs[self.templateIndex]


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
            medComList = []
            for i in range(self.refCount):
                if i == self.templateIndex:
                    continue
                comi = self.refSigs[i][com]
                templateComDis = naive_dtw(comi, self.templateSig[com], PENALIZATION[com], THRESHOLD[com])
                templateComList.append(templateComDis)
                comDisList = []
                for j in range(self.refCount):
                    if i == j:
                        continue
                    comj = self.refSigs[j][com]
                    comDisList.append(naive_dtw(comi, comj, PENALIZATION[com], THRESHOLD[com]))
                maxComList.append(max(comDisList))
                minComList.append(min(comDisList))
                avgComList.append(numpy.mean(comDisList))
                medComList.append(numpy.median(comDisList))
            if "template" in FEATURE_TYPE[com]:
                self.base["template" + com] = numpy.mean(templateComList)
            if "max" in FEATURE_TYPE[com]:
                self.base["max"+com] = numpy.mean(maxComList)
            if "min" in FEATURE_TYPE[com]:
                self.base["min"+com] = numpy.mean(minComList)
            if "avg" in FEATURE_TYPE[com]:
                self.base["avg"+com] = numpy.mean(avgComList)
            if "med" in FEATURE_TYPE[com]:
                self.base["med"+com] = numpy.mean(medComList)
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
            medComDis = numpy.median(comDisList)
            if "template" in FEATURE_TYPE[com]:
                featureVec.append(templateComDis / self.base["template"+com])
            if "max" in FEATURE_TYPE[com]:
                featureVec.append(maxComDis / self.base["max"+com])
            if "min" in FEATURE_TYPE[com]:
                featureVec.append(minComDis / self.base["min"+com])
            if "avg" in FEATURE_TYPE[com]:
                featureVec.append(avgComDis / self.base["avg"+com])
            if "med" in FEATURE_TYPE[com]:
                featureVec.append(medComDis / self.base["med"+com])
            
        return featureVec

class PersonTest(Person):
    def __init__(self, refSigs):
        super(PersonTest, self).__init__(refSigs, None)

class PersonTraining(Person):
    def __init__(self, signatures):
        """
        """
        super(PersonTraining, self).__init__(signatures["genuine"][0:REF_COUNT],
                signatures["genuine"][REF_COUNT:]+signatures["forgery"])
        
        self.genuineSigs = signatures["genuine"][REF_COUNT:]
        self.forgerySigs = signatures["forgery"]

        # LOGGER.info("Reference signature count: %d, test signature count: %d, \
                # genuine test signatures: %d, forgery test signatures: %d" % \
                # (self.refCount, len(self.testSigs), len(self.genuineSigs), len(self.forgerySigs)))
        LOGGER.info("Reference signature count: %d" % self.refCount)
        LOGGER.info("Test signature count: %d" % len(self.testSigs))
        LOGGER.info("Genuine test signature: %d" % len(self.genuineSigs))
        LOGGER.info("Forgery test signature: %d" % len(self.forgerySigs))

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

        self.processor = processor.PreProcessor()

        trainSets, testSets = self.get_signatures_from_susig()
        self.train_set = trainSets
        self.test_set = testSets

        LOGGER.info("Train set count : %d" % len(trainSets))
        LOGGER.info("Test set count : %d" % len(testSets))
        LOGGER.info("Genuine count in train set : %d" % len(trainSets[0]["genuine"]))
        LOGGER.info("Forgery count in train set : %d" % len(trainSets[0]["forgery"]))
        LOGGER.info("Genuine count in test set : %d" % len(testSets[0]["genuine"]))
        LOGGER.info("Forgery count in test set : %d" % len(testSets[0]["forgery"]))

        self.pca = PCA(n_components=1)


        if CLASSIFIER == "SVM":
            self.svm = svm.SVC()
        elif CLASSIFIER == "GBC":
            self.svm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
        elif CLASSIFIER == "RFC":
            # self.svm = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS,
                # max_features = MAX_FEATURES, min_samples_leaf = MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH)
            self.svm = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS)

        if USE_PCA and LinearClassifier == "LogisticRegression":
            self.svm = LogisticRegression()
        elif USE_PCA:
            self.svm = svm.SVC()

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

        genuineCount = len(genuineX)
        trainX = genuineX + forgeryX
        if USE_PCA:
            trainX = self.pca.fit_transform(trainX)

            pcaGenuine = trainX[:genuineCount]
            pcaForgery = trainX[genuineCount:]
            for pcaG in pcaGenuine:
                LOGGER.info("Genuine PCA %s" % pcaG)
            for pcaF in pcaForgery:
                LOGGER.info("Forgery PCA %s" % pcaF)

        trainY = genuineY + forgeryY

        self.svm.fit(trainX, trainY)

    def pre_process_for_signle_signature(self, signature):
        RX = signature[0]
        RY = signature[1]
        P  = signature[2]
        if PRE_PROCESSOR_SWITCH:
            if SIZE_NORM_SWITCH:
                RX, RY = self.processor.size_normalization(RX, RY, 400, 200)
            if LOCAL_NORMAL_TYPE == "mid":
                RX, RY = self.processor.location_normalization(RX, RY)
            elif LOCAL_NORMAL_TYPE == "offset":
                RX, RY = self.processor.offset_to_origin_normalization(RX, RY)
        return [RX, RY, P]

    def reconstructSignature(self, signature):
        """
        Reconstruct signature to dictionary like object
        """
        X = signature[0]
        Y = signature[1]
        P = signature[2]
        VX = self.calculate_delta(X)
        VY = self.calculate_delta(Y)
        VP = self.calculate_delta(P)
        AX = self.calculate_delta(VX)
        AY = self.calculate_delta(VY)
        signature = {"X": X, "Y": Y, "P": P, "VX": VX, "VY": VY, "VP": VP, "AX": AX, "AY": AY}
        return signature

    def get_signatures_from_susig_folder(self, folder):
        
        signatures = {}
        for fileName in os.listdir(folder):
            filePath = os.path.join(folder, fileName)
            with open(filePath) as fp:
                lines = fp.readlines()
                X = []
                Y = []
                T = []
                P = []
                for line in lines[2:]:
                    items = line.split()
                    X.append(float(items[0]))
                    Y.append(float(items[1]))
                    T.append(float(items[2]))
                    P.append(float(items[3]))
                personID = fileName.split(".")[0].split("_")[0]
                if personID not in signatures:
                    signatures[personID] = []
                signature = self.pre_process_for_signle_signature([X, Y, P])
                signature = self.reconstructSignature(signature)
                signatures[personID].append(signature)
        return signatures

    def get_signatures_from_susig(self):
        """
        Load original data from susig
        @return: trainSets, testSets
        """
        LOGGER.info("Getting signatures from susig")
        trainSignatures = {}

        # For train forgery
        trainForgeryFolder = "../data/SUSig/VisualSubCorpus/FORGERY"
        trainForgerySubSigs = self.get_signatures_from_susig_folder(trainForgeryFolder)
        for (personID, value) in trainForgerySubSigs.items():
            if personID not in trainSignatures:
                trainSignatures[personID] = {"genuine": [], "forgery": []}
            trainSignatures[personID]["forgery"].extend(value)

        # For train genuine session1
        trainGenuineSession1Folder = "../data/SUSig/VisualSubCorpus/GENUINE/SESSION1"
        trainGenuineSession1SubSigs= self.get_signatures_from_susig_folder(trainGenuineSession1Folder)
        for (personID, value) in trainGenuineSession1SubSigs.items():
            if personID not in trainSignatures:
                trainSignatures[personID] = {"genuine": [], "forgery": []}
            trainSignatures[personID]["genuine"].extend(value)

        if MULTI_SESSION:
            # For train genuine session2
            trainGenuineSession2Folder = "../data/SUSig/VisualSubCorpus/GENUINE/SESSION2"
            trainGenuineSession2SubSigs = self.get_signatures_from_susig_folder(trainGenuineSession2Folder)
            for (personID, value) in trainGenuineSession2SubSigs.items():
                if personID not in trainSignatures:
                    trainSignatures[personID] = {"genuine": [], "forgery": []}
                trainSignatures[personID]["genuine"].extend(value)

        trainSets = trainSignatures.values()

        # For test sets
        testSignatures = {}
        testForgeryFolder = "../data/SUSig/VisualSubCorpus/VALIDATION/VALIDATION_FORGERY"
        testForgerySubSigs = self.get_signatures_from_susig_folder(testForgeryFolder)
        for (personID, value) in testForgerySubSigs.items():
            if personID not in testSignatures:
                testSignatures[personID] = {"genuine": [], "forgery": []}
            testSignatures[personID]["forgery"].extend(value)

        testGenuineFolder = "../data/SUSig/VisualSubCorpus/VALIDATION/VALIDATION_GENUINE"
        testGenuineSubSigs = self.get_signatures_from_susig_folder(testGenuineFolder)
        for (personID, value) in testGenuineSubSigs.items():
            if personID not in testSignatures:
                testSignatures[personID] = {"genuine": [], "forgery": []}
            testSignatures[personID]["genuine"].extend(value)
        testSets = testSignatures.values()

        """
        print "train set count : %d" % len(trainSets)
        print "test set count : %d" % len(testSets)

        print "train set genuine count: %d" % len(trainSets[0]["genuine"])
        print "train set forgery count: %d" % len(trainSets[0]["forgery"])
        print "test set genuine count: %d" % len(testSets[0]["genuine"])
        print "test set forgery count: %d" % len(testSets[0]["forgery"])
        """
        # return trainSets[:70], trainSets[70:] + testSets
        return testSets+trainSets[0:TRAIN_SET_COUNT-10], trainSets[TRAIN_SET_COUNT-10:]
        # return testSets, trainSets

    def calculate_delta(self, valueList):
        deltaValueList = [valueList[i] - valueList[i-1] for i in range(1, len(valueList))]
        return deltaValueList

    def test(self):
        LOGGER.info("Start test")
        count = 1
        test_set = self.test_set
        if TRAIN_SET_INCLUDE:
            test_set.extend(self.train_set)
        forgery_test_result = []
        genuine_test_result = []
        random_test_result = []

        genuine_test_dis = []
        forgery_test_dis = []

        for i in range(len(test_set)):
            one_test_set = test_set[i]
            LOGGER.info("Test signature: %d" % count)
            count += 1
            personTest = PersonTest(one_test_set["genuine"][0:REF_COUNT])
            genuine_set = one_test_set["genuine"][REF_COUNT:]
            forgery_set = one_test_set["forgery"]
            random_set = []

            for j in range(len(genuine_set)):
                sig = genuine_set[j]
                dis = personTest.calc_dis(sig)
                if USE_PCA:
                    dis = self.pca.transform(dis)
                    genuine_test_dis.append(dis.tolist()[0][0])
                res = self.svm.predict(dis)
                LOGGER.info("Genuine Test: Result: %s, %s" % (res, dis))
                genuine_test_result.append(res)
                if (res != 1):
                    LOGGER.fatal("FalseReject: uid: %d, sid: %d" % (i, j))

            for j in range(len(forgery_set)):
                sig = forgery_set[j]
                dis = personTest.calc_dis(sig)
                if USE_PCA:
                    dis = self.pca.transform(dis)
                    forgery_test_dis.append(dis.tolist()[0][0])
                res = self.svm.predict(dis)
                LOGGER.info("Forgery Test: Result: %s, %s" % (res, dis))
                forgery_test_result.append(res)
                if (res != 0):
                    LOGGER.fatal("FalseAccept: uid: %d, sid: %d" % (i, j))

            if RANDOM_FORGERY_INCLUDE:
                for j in range(len(test_set)):
                    if i == j:
                        continue
                    random_set.extend(test_set[j]["genuine"])
                    random_set.extend(test_set[j]["forgery"])

                # train set included
                for one_train_set in self.train_set:
                    random_set.extend(one_train_set["genuine"])
                    random_set.extend(one_train_set["forgery"])

                for j in range(len(random_set)):
                    sig = random_set[j]
                    dis = personTest.calc_dis(sig)
                    if USE_PCA:
                        dis = self.pca.transform(dis)
                        forgery_test_dis.append(dis.tolist()[0][0])
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


        if USE_PCA:
            # Compute Equal Error Rate
            genuine_test_dis_list = sorted(genuine_test_dis, reverse=True) # desending order
            forgery_test_dis_list = sorted(forgery_test_dis) # asending order

            lastGap = 100.0
            genuineCount = len(genuine_test_dis_list)
            forgeryCount = len(forgery_test_dis_list)
            falseRejectRate = None
            falseAcceptRate = None
            for i in range(genuineCount):
                pivotal = genuine_test_dis_list[i]
                falseRejectRate = float(i) / genuineCount
                j = 0
                while j < forgeryCount and forgery_test_dis_list[j] <= pivotal:
                    j += 1
                falseAcceptRate = float(j) / forgeryCount
                gap = abs(falseAcceptRate - falseRejectRate)
                if gap == 0.0:
                    break;
                elif gap < lastGap:
                    lastGap = gap
                else:
                    break
            LOGGER.info("falseRejectRate: %f, falseAcceptRate: %f" % (falseRejectRate, falseAcceptRate))

            LOGGER.info("TestResultGenuine : %s" % genuine_test_dis)
            LOGGER.info("TestResultForgery : %s" % forgery_test_dis)

def test_DTW():
    start = time.time()
    driver = Driver()
    driver.test()
    end = time.time()
    LOGGER.info("Total time : %f" % (end - start))

if __name__ == "__main__":
    try:
        test_DTW() 
    except Exception as e:
        LOGGER.error("Exception %s" % e)
        LOGGER.exception("message")
