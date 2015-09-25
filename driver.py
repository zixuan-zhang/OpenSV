#! /usr/bin/env python
#coding=utf-8

import sys, os

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

import settings
from processor import PreProcessor, DataProcessor
fileName = 'USER2_23.txt'

def get_data(fileName):
    dataDir = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
    with open(dataDir) as fp:
        lines = fp.readlines()
        X = []
        Y = []
        T = []
        for line in lines[1:]:
            items = line.split()
            X.append(int(items[0]))
            Y.append(int(items[1]))
            T.append(int(items[2]))
    return X, Y, T


def draw_one(fileName):
    X, Y, T = get_data(fileName)

    processor = PreProcessor()
    plt.plot(X, Y)
    plt.figure()

    #X, Y = processor.location_normalization(X, Y)
    #X, Y = processor.size_normalization(X, Y)
    print X
    [X, Y] = processor.gauss_smoothing([X, Y])
    print X
    plt1.plot(X, Y)
    plt.figure()

def data_test(fileName):
    X, Y, T = get_data(fileName)
    dataProcessor = DataProcessor()
    preProcessor = PreProcessor()
    preProcessor.duplicated_point_split(X, Y, T=T)
    R = dataProcessor.radius(X, Y)
    VX = dataProcessor.velocity(X, T)
    VY = dataProcessor.velocity(Y, T)
    AV = dataProcessor.abs_velocity(VX, VY)

if __name__ == "__main__":
    for i in range(1, 5):
        fileName = "USER%d_1.txt" % i
        #draw_one(fileName)
        data_test(fileName)
        print
    

