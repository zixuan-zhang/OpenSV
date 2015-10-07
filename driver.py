#! /usr/bin/env python
#coding=utf-8

import sys, os

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

import settings
from processor import PreProcessor, DataProcessor
from feature_extractor import FeatureExtractor

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

    def __init__(self):
        pass

    def pre_process(self):
