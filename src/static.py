#! /usr/bin/env python
#coding:utf-8

import os
import numpy

import utils

def get_data():
    """
    Load original data from file
    """
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

def calc_points_in_signature():
    signatures = get_data()
    for uid in range(40):
        uSigCounts = []
        for sig in range(40):
            uSigCounts.append(len(signatures[uid][sig][0]))
        genuineCounts = uSigCounts[:20]
        forgeryCounts = uSigCounts[20:]
        print "uid: %d" % uid
        print ">>>Genuine: min: %d, max: %d, avg: %d, std: %d" % (min(genuineCounts),
                max(genuineCounts), numpy.mean(genuineCounts), numpy.std(genuineCounts))
        print ">>>Forgery: min: %d, max: %d, avg: %d, std: %d" % (min(forgeryCounts),
                max(forgeryCounts), numpy.mean(forgeryCounts), numpy.std(forgeryCounts))

if __name__ == "__main__":
    calc_points_in_signature()
