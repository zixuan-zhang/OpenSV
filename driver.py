#! /usr/bin/env python
#coding=utf-8

import sys, os

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

import settings
from processor import PreProcessor, SVMProcessor
from feature_extractor import SVMFeatureExtractor

def get_data(fileName):
    dataDir = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
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

    def get_data_from_file(self, fileName):
        dataDir = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
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

class SVMDriver(Driver):

    def __init__(self, fileName):
        self.fileName = fileName
        self.preProcessor = PreProcessor()
        self.svmProcessor = SVMProcessor()
        self.svmFeatureExtractor = SVMFeatureExtractor()

    def pre_process(self):
        """
            pre-preocoess
        """
        X, Y, T, P = self.get_data_from_file(self.fileName)
        self.preProcessor.duplicated_point_split(X, Y, P, T=T)
        [X, Y, P] = self.preProcessor.gauss_smoothing(X, Y, P) 
        X, Y = self.preProcessor.size_normalization(X, Y)
        X, Y = self.preProcessor.location_normalization(X, Y)

        ListT, ListX, ListY, ListP = self.preProcessor.signature_segmentation(T, X, Y, P)

        return ListT, ListX, ListY, ListP

    def data_process(self):
        ListT, ListX, ListY, ListP = self.pre_process()
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

    def gen_features(self):
        ListT, ListX, ListY, ListR, ListP, ListVX, ListVY, ListVR, ListVP, \
                ListAX, ListAY, ListAR = self.data_process()
        self.svmFeatureExtractor.generate_features(ListT, ListX, ListY, ListR,
                ListP, ListVX, ListVY, ListVR, ListVP, ListAX, ListAY, ListAR)
        self.svmFeatureExtractor.display_features()


if __name__ == "__main__":
    svmDriver = SVMDriver("")
