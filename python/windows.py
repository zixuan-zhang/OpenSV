#! /usr/bin/env python
#coding=utf-8
'''
    @File  : view.py
    @Author: Zhang Zixuan
    @Email : zixuan.zhang.victor@gmail.com
    @Blog  : www.noathinker.com
    @Date  : 2015年10月29日 星期四 09时25分04秒
'''

import os
import time
import numpy
import itertools
import matplotlib.pyplot as plt

import utils
import settings
from driver import AutoEncoderDriver, SimilarityDriver
from processor import PreProcessor


def _input_uid():
    while True:
        print "please input uid. 0-40"
        print ">>>",
        input = raw_input()
        try:
            uid = int(input)
            assert uid >= 0 and uid < 40
            return uid
        except Exception, err:
            print "input illegal, please retype"
            continue

def _input_sid():
    while True:
        print "please input sid. 0-40"
        print ">>>",
        input = raw_input()
        try:
            sid = int(input)
            assert sid >= 0 and sid < 40
            return sid
        except Exception, err:
            print "input illegal, please retype"
            continue

def autoencoder_view_features(loadFromDump=False):
    driver = AutoEncoderDriver()
    if not loadFromDump:
        driver.load_data()
        driver.size_normalization()
        driver.imagize()
        driver.generate_features()
        driver.dump_feature()
    else:
        driver.load_feature()
    while True:
        uid = _input_uid()
        sid = _input_sid()
        print driver.features[uid][sid]

def autoencoder_view_origin():
    driver = AutoEncoderDriver()
    driver.load_data()
    driver.size_normalization()
    while True:
        uid = _input_uid()
        sid = _input_sid()
        plt.plot(driver.data[uid][sid][0],
                driver.data[uid][sid][1])
        plt.show()

def autoencoder_view():
    # use the following 4 steps, you can use iamge data
    driver = AutoEncoderDriver()
    driver.load_data()
    driver.size_normalization()
    driver.imagize()

    while True:
        uid = _input_uid() 
        sid = _input_sid()
        image = driver.data[uid][sid]
        width, height = image.shape
        for i in range(width):
            for j in range(height):
                if image[i][j] != 0:
                    print ".",
                else:
                    print " ",
            print 
        print 

def auto_feature_similarity_view(uid, cnt):
    """
    这个函数显示以uid为sample, 其他笔记与sample的距离
    """
    similarityDriver = SimilarityDriver()
    sampleDis, genuineDis, forgedSigOriDis, forgedSigOthDis = similarityDriver.similarity_distrubution(uid, cnt)

    count = 0
    sampleX = range(count, len(sampleDis) + count)
    count += len(sampleX)
    genuineX = range(count, len(genuineDis) + count)
    count += len(genuineDis)
    forgedSigOriX = range(count, len(forgedSigOriDis) + count)
    count += len(forgedSigOriDis)
    forgedSigOthX = range(count, len(forgedSigOthDis) + count)

    plt.scatter(sampleX, sampleDis, color="green") 
    plt.scatter(genuineX, genuineDis, color="red")
    plt.scatter(forgedSigOriX, forgedSigOriDis, color="blue")
    plt.scatter(forgedSigOthX, forgedSigOthDis, color="black")
    plt.show()

def display_original_signature(uid, sid):
    folder = "../data/Task2"
    fileName = "%s/U%dS%d.TXT" % (folder, uid+1, sid+1)
    X, Y, T, P = utils.get_data_from_file(fileName)
    return X, Y
    # plt.scatter(X, Y)
    # plt.suptitle(fileName)
    # plt.figure(figsize=(3,3))
    # time.sleep(0.3)

def draw_plot(X, Y):
    plt.plot(X, Y)
    plt.show()

def draw_svc():
    for sid in range(40):
        plt.title("%d-genuine" % (sid+1))
        plt.figure(num="%d-genuine" % (sid+1), figsize=(24, 11))
        for gid in range(20):
            X, Y = display_original_signature(sid, gid)
            plt.subplot(3, 7, gid+1)
            plt.title(str(gid+1))
            plt.scatter(X, Y)
        plt.savefig("../data/task2_image/%d_0" % (sid+1))

        plt.figure(num="%d-forgery" % (sid+1), figsize=(24, 11))
        for fid in range(20, 40):
            X, Y = display_original_signature(sid, fid)
            plt.subplot(3, 7, fid-20+1)
            plt.title(str(fid+1))
            plt.scatter(X, Y)
        plt.savefig("../data/task2_image/%d_1" % (sid+1))

def get_x_y_from_self_file(filePath):
    X = []
    Y = []
    with open(filePath) as fp:
        for line in fp:
            if line == '\n':
                continue
            x = float(line.strip().split()[1])
            y = float(line.strip().split()[2])
            X.append(x)
            Y.append(y)
    maxY = max(Y)
    Y = [maxY - y for y in Y]
    return X, Y

def draw_self(_type):
    plt.clf()
    folder = "../data/self/%s" % _type
    filePaths = os.listdir(folder)
    filePaths = sorted(filePaths)
    groups = itertools.groupby(filePaths, lambda x : x.split("_")[0])
    for k, g in groups:
        plt.title(k)
        plt.figure(num=k, figsize=(30,20))
        count = 1
        for v in g:
            # if count > 21:
                # continue
            filePath = os.path.join(folder, v)
            X, Y = get_x_y_from_self_file(filePath)
            p = plt.subplot(7,5, count)
            count += 1
            plt.title(v)
            p.plot(X, Y)
        plt.savefig("../data/self/image/%s/%s" % (_type, k))

class Integer(object):
    def __init__(self):
        self.value = 1

    def value(self):
        print self.value
        return self.value

    def inc(self):
        self.value += 1

    def __str__(self):
        return str(self.value)

def _compare_genuine_forgery(genuineList, forgeryList, count):
    genuineFolder = "../data/self/genuines"
    forgeryFolder = "../data/self/forgeries"
    for fileName in genuineList:
        filePath = os.path.join(genuineFolder, fileName)
        X, Y = get_x_y_from_self_file(filePath)
        p = plt.subplot(2,3,count.value)
        p.cla()
        p.set_xticks([])
        p.set_yticks([])
        count.inc()
        plt.title("Genuine Sample: %s" % fileName)
        p.plot(X,Y, "k")
    for fileName in forgeryList:
        filePath = os.path.join(forgeryFolder, fileName)
        X, Y = get_x_y_from_self_file(filePath)
        p = plt.subplot(2,3,count.value)
        p.set_xticks([])
        p.set_yticks([])
        count.inc()
        plt.title("Forgery Sample: %s" % fileName)
        p.plot(X,Y, "k")

def compare_genuine_forgery():
    genuineList = [
            ["003_001", "003_002", "003_003"],
            # ["015_001", "015_002"],
            # ["011_001", "011_002", "011_003"],
            ]

    forgeryList = [
            ["003_001", "003_002", "003_003"],
            # ["015_001", "015_002"],
            # ["011_015", "011_016", "011_017"],
            ]

    count = Integer()
    plt.figure(num=6, figsize=(12,9))

    for (gList, fList) in zip(genuineList, forgeryList):
        _compare_genuine_forgery(gList, fList, count)

    plt.savefig("../data/%s" % str(int(time.time())))
    plt.show()

def get_data_from_susig_file(fileName):
    preProcessor = PreProcessor()
    print "Getting data from %s" % fileName
    X = []
    Y = []
    P = []
    with open(fileName) as fp:
        lines = fp.readlines()
        for line in lines[2:]:
            items = line.split()
            x = float(items[0])
            y = float(items[1])
            p = float(items[3])
            X.append(x)
            Y.append(y)
            P.append(p)
    X, Y = preProcessor.size_normalization(X,Y,400,200)
    return X, Y, P

def draw_susig():
    plt.ylim([-200,600])
    _id = "012"
    file1 = "../data/susig/SUSig/VisualSubCorpus/VALIDATION/VALIDATION_GENUINE/%s_1_1.sig" % _id
    file2 = "../data/susig/SUSig/VisualSubCorpus/VALIDATION/VALIDATION_GENUINE/%s_1_2.sig" % _id
    file3 = "../data/susig/SUSig/VisualSubCorpus/VALIDATION/VALIDATION_FORGERY/%s_f_6.sig" % _id
    X1,Y1,P1 = get_data_from_susig_file(file1)
    X2,Y2,P2 = get_data_from_susig_file(file2)
    X3,Y3,P3 = get_data_from_susig_file(file3)

    plt.plot(range(len(X1)), X1, "b")
    plt.plot(range(len(X2)), X2, "g")
    #plt.plot(range(len(X3)), X3, "r")
    plt.show()

def get_data_from_self_file(fileName):
    T = []
    X = []
    Y = []
    P = []
    with open(fileName) as fp:
        lines = fp.readlines()
        for line in lines:
            if not line.strip():
                continue
            items = line.strip().split()
            T.append(float(items[0]))
            X.append(float(items[1]))
            Y.append(float(items[2]))
            P.append(float(items[3]))
    if len(T) > 0:
        T = [t - T[0] for t in T]
    Y = [max(Y)- y for y in Y]
    return T, X, Y, P

def compare_self_x_axis():
    file1 = "/home/zixuan/workspace/OpenSV/data/self/genuines/001_001"
    file2 = "/home/zixuan/workspace/OpenSV/data/self/genuines/001_002"
    file3 = "/home/zixuan/workspace/OpenSV/data/self/forgeries/001_010"

    T1, X1, Y1, P1 = get_data_from_self_file(file1)
    T2, X2, Y2, P2 = get_data_from_self_file(file2)
    T3, X3, Y3, P3 = get_data_from_self_file(file3)


    fig, ax = plt.subplots()
    ax.plot(range(len(X1)), X1, "k--", label="Genuine Signature 1")
    ax.plot(range(len(X2)), X2, "k:", label="Genuine Signature 2")
    ax.plot(range(len(X3)), X3, "k", label="Forgery Signature")

    legend = ax.legend(loc="upper left", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")

    plt.xlabel("Time")
    plt.ylabel("X axis")

    plt.show()

def compare_self_vx_axis():
    file1 = "/home/zixuan/workspace/OpenSV/data/self/genuines/001_001"
    file2 = "/home/zixuan/workspace/OpenSV/data/self/genuines/001_002"
    file3 = "/home/zixuan/workspace/OpenSV/data/self/forgeries/001_010"

    T1, X1, Y1, P1 = get_data_from_self_file(file1)
    T2, X2, Y2, P2 = get_data_from_self_file(file2)
    T3, X3, Y3, P3 = get_data_from_self_file(file3)

    VX1 = [X1[i]-X1[i-1] for i in range(1, len(X1))]
    VX2 = [X2[i]-X2[i-1] for i in range(1, len(X2))]
    VX3 = [X3[i]-X3[i-1] for i in range(1, len(X3))]

    fig, ax = plt.subplots()
    ax.plot(range(len(VX1)), VX1, "k", label="Genuine Signature 1")
    ax.plot(range(len(VX2)), VX2, "k--", label="Genuine Signature 2")
    ax.plot(range(len(VX3)), VX3, "k:", label="Forgery Signature")

    legend = ax.legend(loc="upper left", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")

    plt.xlabel("Time")
    plt.ylabel("VX axis")

    plt.show()

def stroke_segmentation(T, X, Y):
    """
        Stroke segmentation.
        Ensure type in T is double.
    """
    assert(len(T) == len(X) and len(T) == len(Y))
    timeSlots = []
    for i in range(len(T) - 1):
        timeSlots.append(T[i+1] - T[i])

    medianSlot = numpy.median(timeSlots)
    # meanSlot = numpy.mean(timeSlots)

    slot = medianSlot

    strokes = []
    stroke = ([], [], [])
    if len(T) > 1:
        stroke[0].append(T[0])
        stroke[1].append(X[0])
        stroke[2].append(Y[0])
    for i in range(1, len(T)):
        if T[i] - T[i-1] > slot and stroke:
            strokes.append(stroke)
            stroke = ([], [], [])
        stroke[0].append(T[i])
        stroke[1].append(X[i])
        stroke[2].append(Y[i])

    if stroke:
        strokes.append(stroke)
    return strokes

def test_stroke_segment():
    fileName = "/home/zixuan/workspace/OpenSV/data/self/genuines/005_000"
    T, X, Y, P = get_data_from_self_file(fileName)

    strokes = stroke_segmentation(T, X, Y)
    print "stoke count : %d" % len(strokes)

    for stroke in strokes:
        if not stroke:
            continue
        (X, Y) = (stroke[1], stroke[2])
        plt.plot(X, Y)

    # plt.plot(X, Y)
    plt.show()

def test():
    fileName = "/home/zixuan/workspace/OpenSV/data/self/genuines/001_000"
    X, Y = get_x_y_from_self_file(fileName)
    maxY = max(Y)
    minY = min(Y)
    Y = [maxY - y for y in Y]
    assert(len(X) == len(Y))
    draw_plot(X, Y)
    plt.show()

if __name__ == "__main__":
    # draw_self("forgeries")
    # draw_self("genuines")
    #test()
    # draw_susig()
    # compare_self_x_axis()
    # compare_self_vx_axis()
    # compare_genuine_forgery()
    test_stroke_segment()
