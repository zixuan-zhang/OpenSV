#! /usr/bin/env python
#coding=utf-8

import sys, os
import numpy
import json
import math
import scipy
import copy

from sklearn import svm
from sklearn import tree

import settings
from processor import PreProcessor, SVMProcessor
from feature_extractor import SVMFeatureExtractor, ProbFeatureExtractor,\
        AutoEncoderFeatureExtractor

class Driver(object):

    def get_data_from_file(self, filePath):
        #dataDir = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
        with open(filePath) as fp:
            lines = fp.readlines()
            X = []
            Y = []
            T = []
            P = []
            for line in lines[1:]:
                items = line.split()
                X.append(float(items[0]))
                Y.append(float(items[1]))
                T.append(int(items[2]))
                P.append(float(items[6]))
        return X, Y, T, P

    def pre_process(self, fileName):
        """
            pre-preocoess
        """
        X, Y, T, P = self.get_data_from_file(fileName)
        self.preProcessor.duplicated_point_split(X, Y, P, T=T)
        [X, Y, P] = self.preProcessor.gauss_smoothing(X, Y, P) 
        X, Y = self.preProcessor.size_normalization(X, Y)
        X, Y = self.preProcessor.location_normalization(X, Y)

        ListT, ListX, ListY, ListP = self.preProcessor.signature_segmentation(T, X, Y, P)

        return ListT, ListX, ListY, ListP

    def data_process(self, fileName):
        pass

    def generate_features(self, fileName):
        pass

class SVMDriver(Driver):

    def __init__(self):
        self.preProcessor = PreProcessor()
        self.svmProcessor = SVMProcessor()
        self.svmFeatureExtractor = SVMFeatureExtractor()


    def data_process(self, fileName):
        ListT, ListX, ListY, ListP = self.pre_process(fileName)
        ListR = self.svmProcessor.radius(ListX, ListY)
        ListVX = self.svmProcessor.velocity_of_x(ListX, ListT)
        ListVY = self.svmProcessor.velocity_of_y(ListY, ListT)
        ListVR = self.svmProcessor.velocity_of_r(ListR, ListT)
        ListVP = self.svmProcessor.velocity_of_p(ListP, ListT)
        ListAX = self.svmProcessor.acc_of_vx(ListVX, ListT)
        ListAY = self.svmProcessor.acc_of_vy(ListVY, ListT)
        ListAR = self.svmProcessor.acc_of_vr(ListVR, ListT)

        return ListT, ListX, ListY, ListP, ListR, ListVX, ListVY, ListVR, \
                ListVP, ListAX, ListAY, ListAR

    def generate_features(self, fileName):
        ListT, ListX, ListY, ListR, ListP, ListVX, ListVY, ListVR, ListVP, \
                ListAX, ListAY, ListAR = self.data_process(fileName)
        self.svmFeatureExtractor.generate_features(ListT, ListX, ListY, ListR,
                ListP, ListVX, ListVY, ListVR, ListVP, ListAX, ListAY, ListAR)
        return self.svmFeatureExtractor.features()

    def feature_clear(self):
        self.svmFeatureExtractor.clear()

class ProbDriver(Driver):

    def __init__(self):
        self.svmProcessor = SVMProcessor()
        self.probFeatureExtractor = ProbFeatureExtractor()
        self.preProcessor = PreProcessor()

    def data_process(self, filePath):
        ListT, ListX, ListY, ListP = self.pre_process(filePath)
        ListR = self.svmProcessor.radius(ListX, ListY)
        ListVX = self.svmProcessor.velocity_of_x(ListX, ListT)
        ListVY = self.svmProcessor.velocity_of_y(ListY, ListT)
        ListAX = self.svmProcessor.acc_of_vx(ListVX, ListT)
        ListAY = self.svmProcessor.acc_of_vy(ListVY, ListT)

        return ListT, ListR, ListVX, ListVY, ListAX, ListAY

    def generate_features(self, filePath):
        ListT, ListR, ListVX, ListVY, ListAX, ListAY = self.data_process(filePath)
        self.probFeatureExtractor.generate_features(ListT, ListR, ListVX, ListVY, ListAX, ListAY)
        return self.probFeatureExtractor.features()

    def feature_clear(self):
        self.probFeatureExtractor.clear()

    def PS(self, F):
        P = []
        for i in range(len(F)):
            p = numpy.exp(-1 * pow((F[i] - self.meanF[i]), 2) / 2 / self.varF[i]) / numpy.sqrt(2 * self.meanF[i] * self.varF[i])
            P.append(p)
        # print [round(p, 11) for p in P]
        # print numpy.isnan(P)

        PArray = numpy.asarray(P)
        PArray[numpy.isnan(PArray)] = 0

        return PArray.sum()

    def ps_temp(self, ListF):
        self.meanF = []
        self.varF = []

        Features = []
        for F in ListF:
            for i in range(len(F)):
                if i >= len(Features):
                    Features.append([])
                Features[i].append(F[i])
        for F in Features:
            self.meanF.append(numpy.mean(F))
            self.varF.append(numpy.var(F))

class AutoEncoderFeatureDriver(Driver):

    def __init__(self):

        self.data = []
        self.features = []

    def load_feature(self, fileDir = None):
        if not fileDir:
            fileDir = "../data/auto_features"
        os.chdir(fileDir)
        for uid in range(40):
            uidFeatures = []
            for sid in range(40):
                fileName = "u%ds%d.txt" % (uid, sid)
                feature = numpy.loadtxt(fileName)
                uidFeatures.append(feature)
            self.features.append(uidFeatures)

class AutoEncoderDriver(AutoEncoderFeatureDriver):

    def __init__(self):
        # data structure
        self.data = []
        self.features = []
        self.width = 30
        self.height = 30
        self.featureExtractor = AutoEncoderFeatureExtractor(self.width, self.height)
        self.processor = PreProcessor()

        self.driver = ProbDriver()

    def load_data(self):
        dataDir = "../data/Task2"
        os.chdir(dataDir)
        curDir = os.getcwd()
        self.data = []
        for uid in range(1, settings.USER_COUNT+1):
            uidData = []
            for sid in range(1, 41):
                fileName = "U%dS%d.TXT" % (uid, sid)
                X, Y, T, P = self.get_data_from_file(fileName)
                uidData.append((X, Y))
            self.data.append(uidData)
        os.chdir("../..")

    def size_normalization(self):
        data = []
        for uid in range(40):
            uidData = []
            for sid in range(40):
                X, Y = self.processor.size_normalization(
                        self.data[uid][sid][0],
                        self.data[uid][sid][1],
                        self.width, self.height)
                uidData.append((X, Y))
            data.append(uidData)
        self.data = data

    def imagize(self):
        data = []
        for uid in range(40):
            uidData = []
            for sid in range(40):
                image = self.featureExtractor.imagize(self.data[uid][sid][0],
                        self.data[uid][sid][1])
                uidData.append(image)
            data.append(uidData)
        self.data = data

    def train(self, layer_sizes=[500, 300, 100, 50, 20, 10], epoch=20):
        if not self.data:
            self.imagize()

        train_set_x = numpy.asarray(self.data)
        (uCnt, sCnt, pCnt) = train_set_x.shape
        train_set_x = train_set_x.reshape((uCnt*sCnt, pCnt))

        n_ins = (self.width + 1) * (self.height + 1)
        # train data
        self.featureExtractor.train(train_set_x, n_ins, layer_sizes, epoch)

    def generate_features(self):
        """
        generate feature from image to features using stacked autoencoder 
        """

        # train data first
        self.train()

        self.features = []
        for uid in range(40):
            uidFeatures = []
            for sid in range(40):
                feature = self.featureExtractor.generate_features(
                        self.data[uid][sid])
                uidFeatures.append(feature)
                print ">>>uid: %d, sid: %d ends" % (uid, sid)
            self.features.append(uidFeatures)

    def dump_feature(self):
        print "... dumpint features"
        dataDir = "./data"
        os.chdir(dataDir)
        autoFeatureDir = "auto_features"
        if not os.path.exists(autoFeatureDir):
            os.mkdir(autoFeatureDir)
        os.chdir(autoFeatureDir)
        for uid in range(40):
            for sid in range(40):
                fileName = "u%ds%d.txt" % (uid, sid)
                numpy.savetxt(fileName, self.features[uid][sid],fmt="%10.5f")
        os.chdir("../..")


    def train_test_set(self, uid, cnt):
        uidFeatures = self.features[uid]
        train_set_x = []
        pos_set_x = []
        neg_set_x_ori = []
        neg_set_x_oth = []
        for sid in range(cnt):
            train_set_x.append(uidFeatures[sid].tolist())
        for sid in range(cnt, 20):
            pos_set_x.append(uidFeatures[sid].tolist())
        for sid in range(20, 40):
            neg_set_x_ori.append(uidFeatures[sid].tolist())
        for i in range(40):
            if i == uid:
                continue
            for sid in range(40):
                neg_set_x_oth.append(self.features[i][sid].tolist())

        return train_set_x, pos_set_x, neg_set_x_ori, neg_set_x_oth

    def score_of_uid(self, uid, cnt):

        train_set_x, pos_set_x, neg_set_x_ori, neg_set_x_oth = self.train_test_set(uid, cnt)

        driver = ProbDriver()
        # print ">>> training..."
        driver.ps_temp(train_set_x)

        # print ">>> train set"
        trainPS = []
        for X in train_set_x:
            ps = driver.PS(X)
            # print ps
            trainPS.append(ps)
        threshold = min(trainPS)
        # print ">>> train set min is ", threshold

        def _score_of_set(set_x, pos=True):
            size = len(set_x)
            setPS = []
            for X in set_x:
                ps = driver.PS(X)
                setPS.append(ps)
            if pos:
                correctSize = len([ps for ps in setPS if ps >= threshold])
            else:
                correctSize = len([ps for ps in setPS if ps <= threshold])
            return correctSize / float(size)

        # testing process
        # print ">>> postive test set"
        scoreOfPos = _score_of_set(pos_set_x, pos=True)
        # print ">>> total postive set %d, greater than threshold %f" % (len(pos_set_x), scoreOfPos)

        # print ">>> negtive test set"
        scoreOfNegOri = _score_of_set(neg_set_x_ori, pos=False)
        # print ">>> original negtive set %d, less than threshold %f" % (len(neg_set_x_ori), scoreOfNegOri)

        # print ">>> other negtive test set"
        scoreOfNegOth = _score_of_set(neg_set_x_oth, pos=False)
        # print ">>> total negtive set %d, less than threhold %f" % (len(neg_set_x_oth), scoreOfNegOth)

        return scoreOfPos, scoreOfNegOri, scoreOfNegOth

    def score(self):
        self.load_feature()
        scoreOfPos = []
        scoreOfNegOri = []
        scoreOfNegOth = []
        for cnt in [3, 5, 7, 10, 15]:
            for uid in range(40):
                pos, negOri, negOth = self.score_of_uid(uid, cnt)
                scoreOfPos.append(pos)
                scoreOfNegOri.append(negOri)
                scoreOfNegOth.append(negOth)
            print numpy.mean(scoreOfPos), numpy.mean(scoreOfNegOri), numpy.mean(scoreOfNegOth)

class JaccardDriver(Driver):
    """
    A similarity method
    """

    def __init__(self, train_set, threshold=0.7):
        if not train_set:
            return
        self.train_set = train_set
        self.threshold = threshold
        self.featureSize = len(train_set[0])
        self.trainCount = len(train_set)

        self.genuineFeature = [0] * self.featureSize

        for X in train_set:
            for i in range(self.featureSize):
                x = round(X[i])
                self.genuineFeature[i] += x

        for i in range(self.featureSize):
            self.genuineFeature[i] = round(float(self.genuineFeature[i]) /
                self.trainCount)

    def similarity(self, test_set, threshold=None):
        """
        求test_set与train过的genuineFeature的相似性
        """
        if not threshold:
            threshold = self.threshold
        assert len(test_set) == self.featureSize

        newTestSet = [round(x) for x in test_set]
        res = sum([1 if newTestSet[i] == self.genuineFeature[i] else 0 for i in range(self.featureSize)])
        return float(res) / self.featureSize 

class SimilarityDriver(AutoEncoderFeatureDriver):
    """
    This class define a few methods to calculate similarity
    """

    def euclid_distance(self, X, Y):
        assert len(X) == len(Y)
        return scipy.spatial.distance.euclidean(X, Y)

    def cos_distance(self, X, Y):
        assert len(X) == len(Y)
        partUp = 0.0
        xSqr = 0.0
        ySqr = 0.0
        for x, y in zip(X, Y):
            partUp += x * y
            xSqr += x ** 2
            ySqr += y ** 2
        partDown = math.sqrt(xSqr * ySqr)
        if partDown == 0:
            return None
        else:
            return partUp / partDown

    def sample_avg(self, samples):
        _sum = copy.deepcopy(samples[0]) # Here needs deep copy
        for i in range(1, len(samples)):
            for j in range(len(_sum)):
                _sum[j] += samples[i][j]
        for i in range(len(_sum)):
            _sum[i] /= len(samples)
        return _sum

    def avg_similarity(self, compares):
        """
        这种求相似度方法先求出samples的平均值，然后compares与avg求相似度distance
        """
        distances = []
        for compare in compares:
            distance = self.euclid_distance(self.avgFeatures, compare)
            if distance == 0.0:
                print "same"
                print self.avgFeatures
                print compare
            distances.append(distance)
        return distances

    def group_similarity(self, samples, compares):
        """
        samples和compares均为多个样本集。本函数将计算sample * compares组合
        的distance
        """
        distances = []

        for sample in samples:
            for compare in compares:
                # distance = self.cos_distance(sample, compare)
                distance = self.euclid_distance(sample, compare)
                distances.append(distance)
        return distances

    def similarity_distrubution(self, uid, sCnt):
        """
        三种签名：
            1. 正确的签名(Genuine Signatures)
            2. 伪造的签名(Forged Signatures)
            3. 普通签名(Regular Signatures)

        假设已经确定的正确的签名为sample，现在需要求出其他Genuine Signature、
            Forged Signatures、Regular Signatures 与sample的距离。
        其中这四类都为集合，所以得出的距离也是list类型，其中是每个待检测签名
            与sample的距离的集合
        """

        if not self.features:
            self.load_feature()
        sample = self.features[uid][0:sCnt]

        genuineSigs = self.features[uid][sCnt:20]
        forgedSigOri = self.features[uid][20:40]
        forgedSigOth = []
        for u in range(40):
            if uid != u:
                forgedSigOth.extend(self.features[u])

        similarity = "avg"
        if "group" == similarity:
            sampleDis = self.group_similarity(sample, sample)
            genuineDis = self.group_similarity(sample, genuineSigs)
            forgedSigOriDis = self.group_similarity(sample, forgedSigOri)
            forgedSigOthDis = self.group_similarity(sample, forgedSigOth)
        elif "avg" == similarity:
            self.avgFeatures = self.sample_avg(sample)
            sampleDis = self.avg_similarity(sample)
            genuineDis = self.avg_similarity(genuineSigs)
            forgedSigOriDis = self.avg_similarity(forgedSigOri)
            forgedSigOthDis = self.avg_similarity(forgedSigOth)

        return sampleDis, genuineDis, forgedSigOriDis, forgedSigOthDis

class SimilarityProbDriver(AutoEncoderFeatureDriver):

    """
    这个类的思想借鉴Probablistic Model.
        1. 根据train features求出 average feature
        2. 每个train feature求出根average feature的距离
        3. 求出上面距离最大的那个，然后乘以rate作为阈值
        3. test feature与average feature求距离，如果小于阈值则为正，反之为负

    Usage:
        driver = SimilarityProbDriver()
        driver.train(trainFeatureList)
        result = driver.test(testFeatureList)
    """

    def __init__(self):
        AutoEncoderFeatureDriver.__init__(self)
        self.rate = 1.5 # mean the times of max distance of training example

    def _euclid_distance(self, X, Y):
        assert len(X) == len(Y)
        return scipy.spatial.distance.euclidean(X, Y)

    def _sample_avg(self, samples):
        _sum = copy.deepcopy(samples[0]) # Here need deep copy
        for i in range(1, len(samples)):
            for j in range(len(_sum)):
                _sum[j] += samples[i][j]
        for i in range(len(_sum)):
            _sum[i] /= len(samples)
        return _sum

    def train(self, trainFeatures):
        self.distanceThreshold= None

        self.featureAvg = self._sample_avg(trainFeatures)
        trainDis = []
        for features in trainFeatures:
            distance = self._euclid_distance(self.featureAvg, features)
            trainDis.append(distance)
        maxDis = max(trainDis)
        self.distanceThreshold = maxDis * self.rate

    def predict(self, testFeatures):
        results = []
        for features in testFeatures:
            distance = self._euclid_distance(self.featureAvg, features)
            result = True if distance <= self.distanceThreshold else False
            results.append(result)
        return results

    def one_user_test(self, uid=0, sCnt=5):

        if not self.features:
            self.load_feature()
        sample = self.features[uid][0:sCnt]

        genuineSigs = self.features[uid][sCnt:20]
        forgedSigOri = self.features[uid][20:40]
        forgedSigOth = []
        for u in range(40):
            if uid != u:
                forgedSigOth.extend(self.features[u])

        # training process
        self.train(sample)

        # test process
        results = self.predict(genuineSigs)
        gscore = sum([1 if x == True else 0 for x in results]) / float(len(genuineSigs))

        results = self.predict(forgedSigOri)
        fscore = sum([1 if x == True else 0 for x in results]) / float(len(forgedSigOri))

        results = self.predict(forgedSigOth)
        rscore = sum([1 if x == False else 0 for x in results]) / float(len(forgedSigOth))

        print "gscore is %f, fscore is %f, rscore is %f" % (gscore, fscore, rscore)
        return gscore, fscore, rscore

    def test(self):
        gscores = []
        fscores = []
        rscores = []
        for uid in range(40):
            gscore, fscore, rscore = self.one_user_test(uid, 5)
            gscores.append(gscore)
            fscores.append(fscore)
            rscores.append(rscore)
        print "final score"
        print "gscore is %f, fscore is %f, rscore is %f" % (numpy.mean(gscores),
                numpy.mean(fscores), numpy.mean(rscores))

def get_training_data(svmDriver):
    print "loading training data"
    training_dir = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR])
    files = os.listdir(training_dir)
    files = sorted(files)
    X = []
    Y = []
    for fileName in files:
        print "processing %s" % fileName
        filePath = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
        svmDriver.feature_clear()
        f = svmDriver.generate_features(filePath)
        X.append(f)
        if fileName.find("_0") != -1 or fileName.find("_1") != -1:
            Y.append(1)
        if fileName.find("_2") != -1 or fileName.find("_3") != -1:
            Y.append(0)
    return X, Y

def get_test_data(svmDriver):
    print "loading test data..."
    test_dir = '/'.join([os.getcwd(), settings.TEST_DATA_DIR])
    files = os.listdir(test_dir)
    files = sorted(files)
    X = []
    for fileName in files:
        print "processing %s" % fileName
        filePath = '/'.join([os.getcwd(), settings.TEST_DATA_DIR, fileName])
        svmDriver.feature_clear()
        f = svmDriver.generate_features(filePath)
        X.append(f)
    return X

def test_auto_driver():
    """
    """
    autoDriver = AutoEncoderDriver()
    autoDriver.score()

def test_jaccard_driver():
    """
    使用Stacked AutoEncoder提取出来的特征，使用JaccardDriver的
    方法看下效果
    """
    autoDriver = AutoEncoderDriver()
    autoDriver.load_feature()

    train_set = autoDriver.features[0][0:20]
    test_set_gen = autoDriver.features[0][0:20]
    test_set_for = autoDriver.features[0][20:40]
    test_set_oth = autoDriver.features[1][0:20]
    test_set_oth1 = autoDriver.features[2][0:20]

    # train process
    jaccardDriver = JaccardDriver(train_set)

    results = []
    for one_test_set in test_set_gen:
        res = jaccardDriver.similarity(one_test_set)
        results.append(res)
    print numpy.mean(results)

    results = []
    for one_test_set in test_set_for:
        res = jaccardDriver.similarity(one_test_set)
        results.append(res)
    print numpy.mean(results)

    results = []
    for one_test_set in test_set_oth:
        res = jaccardDriver.similarity(one_test_set)
        results.append(res)
    print numpy.mean(results)

    results = []
    for one_test_set in test_set_oth1:
        res = jaccardDriver.similarity(one_test_set)
        results.append(res)
    print numpy.mean(results)

def test_similarity_prob_driver():
    """
    测试similarity driver，获得认证效果
    """
    similarityDriver = SimilarityDriver()


if __name__ == "__main__":
    # test_auto_driver()
    # test_jaccard_driver()
    driver = SimilarityProbDriver()
    driver.test()

    """
    svmDriver = ProbDriver()
    XTraining, YTraining = get_training_data(svmDriver)
    XTest = get_test_data(svmDriver)
    svmDriver.ps_temp(XTraining)
    for X in XTraining:
        ps = svmDriver.PS(X)
        print ps
    print
    for X in XTest:
        ps = svmDriver.PS(X)
        print ps
    """
    
    """
    clf = svm.SVC()
    #clf = tree.DecisionTreeClassifier()
    clf.fit(XTraining, YTraining)
    result = clf.predict(XTest)
    print YTraining
    print result
    """
