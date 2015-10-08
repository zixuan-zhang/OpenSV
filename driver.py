#! /usr/bin/env python
#coding=utf-8

import sys, os

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from sklearn import svm
from sklearn import tree

import settings
from processor import PreProcessor, SVMProcessor
from feature_extractor import SVMFeatureExtractor

def get_data(fileDir):
    #dataDir = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
    with open(dataDir) as fp:
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

class Driver(object):

    def __init__(self):
        pass

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

class SVMDriver(Driver):

    def __init__(self):
        self.preProcessor = PreProcessor()
        self.svmProcessor = SVMProcessor()
        self.svmFeatureExtractor = SVMFeatureExtractor()

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

if __name__ == "__main__":
    svmDriver = SVMDriver()
    XTraining, YTraining = get_training_data(svmDriver)
    XTest = get_test_data(svmDriver)
    #clf = svm.SVC()
    clf = tree.DecisionTreeClassifier()
    clf.fit(XTraining, YTraining)
    result = clf.predict(XTest)
    print YTraining
    print result
