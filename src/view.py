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
import matplotlib.pyplot as plt

import settings
from driver import get_data_from_file, AutoEncoderDriver

def _input_uid():
    while True:
        print "please input uid. 0-40"
        print ">>>"
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
        print ">>>"
        input = raw_input()
        try:
            sid = int(input)
            assert sid >= 0 and sid < 40
            return sid
        except Exception, err:
            print "input illegal, please retype"
            continue

def autoencoder_view():
    driver = AutoEncoderDriver()
    driver.load_data()
    driver.imagize()

    while True:
        uid = _input_uid() 
        sid = _input_sid()
        image = driver.data[uid][sid]
        X = []
        Y = []
        for x in len(image):
            row = image[x]
            for y in len(row):
                if image[x][y] != 0:
                    X.append(x)
                    Y.append(y)
        plt.plot(X, Y)
        plt.show()

def draw_plot(X, Y):
    plt.plot(X, Y)
    plt.show()

if __name__ == "__main__":
    fileName = "USER1_1.txt"
    filePath = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
    X, Y, T, P = get_data_from_file(filePath)
    draw_plot(X, Y)
