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
from driver import AutoEncoderDriver

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

def autoencoder_view_features():
    driver = AutoEncoderDriver()
    driver.load_data()
    driver.size_normalization()
    driver.imagize()
    driver.generate_features()
    driver.dump_feature()
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

def draw_plot(X, Y):
    plt.plot(X, Y)
    plt.show()

if __name__ == "__main__":
    autoencoder_view_features()
