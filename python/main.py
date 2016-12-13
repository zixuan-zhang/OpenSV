#coding:utf-8

import time

import settings

from driver import ClassifyDriver, RegressionDriver
from susig_config import SUSIGConfig
from self_config import SelfConfig

def test_DTW():
    # config = SUSIGConfig()
    config = SelfConfig()
    try:
        start = time.time()
        driver = None
        if config.ClassifyOrRegression == settings.Classify:
            driver = ClassifyDriver(config)
        else:
            driver = RegressionDriver(config)
        driver.test()
        end = time.time()
        config.logger.info("Total time : %f" % (end - start))
    except Exception as e:
        config.logger.error("Exception %s" % e)
        config.logger.exception("message")

if __name__ == "__main__":
    test_DTW()
