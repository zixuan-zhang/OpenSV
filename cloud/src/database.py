# encoding: utf-8

##############################################################################
####   File         : database.py
####   Author       : zixuan.zhang.victor@gmail.com
####   Create Date  : 2017年03月18日 星期六 18时40分52秒
##############################################################################

from pymongo import MongoClient

class SignatureStorage(object):
    def __init__(self, config):
        self.config = config
        client = MongoClient(self.config.DatabaseIP, self.config.DatabasePort)
        db = client[self.config.DatabaseName]
        self.table = db[self.config.DatabaseTableName]

    def save(self, _id, signatures):
        self.table.find_one_and_update({"_id": _id}, {"$set": {"signatures": signatures}}, upsert=True)

    def load(self, _id):
        cursor = self.table.find({"_id": _id})
        if cursor.count() == 0:
            return None
        signatures = cursor[0]["signatures"]
        return signatures

def unit_test():
    def mock_config():
        class Config():
            def __init__(self):
                self.DatabaseIP = "127.0.0.1"
                self.DatabasePort = 27017
                self.DatabaseName = "test"
                self.DatabaseTableName = "test"
        return Config()
    config = mock_config()
    storage = SignatureStorage(config)

    assert storage.load("123") == None
    storage.save("123", "456")
    assert storage.load("123") == "456"
    storage.save("123", "789")
    assert storage.load("123") == "789"

if __name__ == "__main__":
    unit_test()
