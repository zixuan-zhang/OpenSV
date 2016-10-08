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
    draw_self("forgeries")
    # test()
