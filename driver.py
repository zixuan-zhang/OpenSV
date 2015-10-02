#! /usr/bin/env python
#coding=utf-8

import sys, os

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

import settings
from processor import PreProcessor, DataProcessor
from feature_extractor import FeatureExtractor
fileName = 'USER2_23.txt'

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


def draw_one(fileName):
    X, Y, T, P = get_data(fileName)

    processor = PreProcessor()
    plt.plot(X, Y)
    plt.figure()

    #X, Y = processor.location_normalization(X, Y)
    #X, Y = processor.size_normalization(X, Y)
    #[X, Y] = processor.gauss_smoothing([X, Y])
    plt1.plot(X, Y)
    plt.figure()

def data_test(fileName):
    X, Y, T, P = get_data(fileName)
    preProcessor = PreProcessor()
    preProcessor.duplicated_point_split(X, Y, T=T)
    segT, segX, segY, segP = preProcessor.signature_segmentation(T, X, Y, P)
    print len(segT), len(segX), len(segY), len(segP)

def dct_test(fileName):
    X, Y, T, P = get_data(fileName)
    dataProcessor = DataProcessor()
    preProcessor = PreProcessor()
    preProcessor.duplicated_point_split(X, Y, T=T)
    R = dataProcessor.radius(X, Y)
    VX = dataProcessor.velocity_of_x(X, T)
    VY = dataProcessor.velocity_of_y(Y, T)
    AV = dataProcessor.abs_velocity(VX, VY)

    extractor = FeatureExtractor()
    DCT_X = extractor.dct(X)
    print len(DCT_X)
    print DCT_X
    print sum(X)
    print sum(DCT_X)

def test1():
    for i in range(1, 2):
        fileName = "USER%d_1.txt" % i

def test2():
    for user in range(1, 5):
        for samp in range(1, 41):
            fileName = 'USER%d_%d.txt' % (user, samp)
            print "%d\t" % samp,
            data_test(fileName)
        print ""

if __name__ == "__main__":
    test2()
    

