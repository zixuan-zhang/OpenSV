# encoding: utf-8

##############################################################################
####   File         : database.py
####   Author       : zixuan.zhang.victor@gmail.com
####   Create Date  : 2017年03月18日 星期六 18时40分52秒
##############################################################################

class SignatureStorage(object):
    def __init__(self):
        self.storage = {} 

    def save(self, _id, signatures):
        self.storage[_id] = signatures

    def load(self, _id):
        return self.storage[_id]

    def exist(self, _id):
        return _id in self.storage
