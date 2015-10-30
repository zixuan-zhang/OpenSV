#! /usr/bin/env python
#coding=utf-8
'''
    @File  : view.py
    @Author: Zhang Zixuan
    @Email : zixuan.zhang.victor@gmail.com
    @Blog  : www.noathinker.com
    @Date  : 2015年10月29日 星期四 09时25分04秒
'''

import matplotlib.pyplot as plt

import settings
from driver import get_data_from_file

def draw_plot(X, Y):
    plt.plot(X, Y)
    plt.show()

if __name__ == "__main__":
    fileName = "USER1_1.txt"
    filePath = '/'.join([os.getcwd(), settings.TRAINING_DATA_DIR, fileName])
    X, Y, T, P = get_data_from_file(filePath)
    draw_plot(X, Y)
