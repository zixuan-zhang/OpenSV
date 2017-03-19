#coding:utf-8

import os
import datetime
import logging

import settings

import numpy
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

class SelfConfig(object):
    """
        SUSIG dataset configurations.
        All global configuration variables start with uppercase letters.
    """
    def __init__(self):

        #### Log configurations
        logFolder = "../data/self_log"
        if not os.path.exists(logFolder):
            os.makedirs(logFolder)
        logFileName = datetime.datetime.now().strftime("%Y%m%d%H%M%S.log")
        logFormat = '%(asctime)s %(levelname)s %(name)s %(message)s'
        logging.basicConfig(filename = "%s/%s" % (logFolder, logFileName), level = logging.INFO, format = logFormat)
        self.logger = logging.getLogger()

        #### Mode configuration just indicates use total dataset or partial dataset.
        self.Mode = settings.ModeNormal

        #### Classification and regression configurations.
        self.ClassifyOrRegression = settings.Classify
        self.Classifier = "RFC" # "RFC", "GBC", "SVM", "MLP", "Logi"
        self.Regressor = "LOG" # RFR, LOG, "GBR", PCA, MLP
        self.ModelDumpFilePath = "../data/self_model.dump"

        #### Signal configurations
        self.SigCompList = ["X", "Y", "VX", "VX", "VY"]

        #### Signal weight when selecting template signature
        self.SignalWeight = {
                "X" : 1.,
                "Y" : 1.,
                "VX": 1.,
                "VY": 1.,
                "P" : 1.
                }

        #### Feature settings
        self.FeatureType = {
                "VX": ["template", "min", "avg"],
                "VY": ["template", "min", "avg"],
                "P": ["template", "min", "avg"],
                "X": ["template", "med", "min", "avg"],
                "Y": ["template", "med", "min", "avg"],
                "VP": ["template", "min", "avg"],
                "AX": ["template", "min", "avg"],
                "AY": ["template", "min", "avg"],
                }

        #### Dynamic Time Warping configurations.
        self.DTWMethod = 2
        self.Penalization = {
                "X": 7,
                "Y": 5,
                "VX": 3,
                "VY": 2,
                "P": 2,
                "VP": 2,
                "AX": 3,
                "AY": 3,
                }
        self.Threshold= {
                "X": 8,
                "Y": 6,
                "VX": 0,
                "VY": 1,
                "P": 2,
                "VP": 0,
                "AX": 4,
                "AY": 3,
                }

        #### Dataset settings
        self.TrainCountRate = 0.7
        self.Dataset = settings.Self
        self.TrainSetCount = 10
        self.RefCount = 5
        self.MultiSession = True

        self.GenuineFolder = "../data/self/GENUINE"
        self.ForgeryFolder = "../data/self/FORGERY"

        #### Preprocessing configurations.
        self.PreProcessorSwitch = True
        self.LocalNormalType = "mid"
        self.RandomForgeryInclude = False
        self.TrainSetInclude = False
        self.SizeNormSwitch = True

        #### Random Forest Tree settings
        self.RFTMaxDepth = 3
        self.RFTMaxFeatures = None
        self.RFTNumEstimators = 200
        self.RFTMinSamplesLeaf = 1
        self.RFTNumJobs = 1

        self.model = None

        if self.ClassifyOrRegression == settings.Classify:
            if self.Classifier == "SVM":
                self.model= svm.SVC()
            elif self.Classifier == "GBC":
                self.model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
            elif self.Classifier == "RFC":
                self.model = RandomForestClassifier(n_estimators=self.RFTNumEstimators, n_jobs=self.RFTNumJobs)
            else:
                raise Exception("Classifier %s not supported" % self.Classifier)
        else:
            if self.Regressor == "LOG":
                self.model = LogisticRegression()
            elif self.Regressor == "RFR":
                self.model = RandomForestRegressor(n_estimators=self.RFTNumEstimators, n_jobs=self.RFTNumJobs)
            elif self.Regressor == "GBR":
                self.model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
            elif self.Regressor == "PCA":
                self.model = PCA(n_components=1)
            else:
                raise Exception("Regressor: %s not supported." % self.Regressor)

        self._print_log()
    def _print_log(self):
        """
            @Description: This function just print the configurations to log.
        """

        self.logger.info("Mode: %s" % self.Mode)

        self.logger.info("Signal List: %s" % self.SigCompList)
        self.logger.info("FeatureType: %s" % self.FeatureType)

        self.logger.info("Method: %d" % self.DTWMethod)
        self.logger.info("PENALIZATION: %s" % self.Penalization)
        self.logger.info("THRESHOLD: %s" % self.Threshold)

        self.logger.info("TrainSetCount: %s" % self.TrainSetCount)
        self.logger.info("Reference Count: %d" % self.RefCount)
        self.logger.info("MultiSession: %s" % self.MultiSession)

        self.logger.info("PreProcessorSwitch: %s" % self.PreProcessorSwitch)
        self.logger.info("LocalNormalizationType: %s" % self.LocalNormalType)
        self.logger.info("RandomForgeryInclude: %s" % self.RandomForgeryInclude)
        self.logger.info("TrainSetInclude: %s" % self.TrainSetInclude)
        self.logger.info("SizeNormSwith: %s" % self.SizeNormSwitch)


        self.logger.info("ClassifyOrRegression: %s" % self.ClassifyOrRegression)
        self.logger.info("ClassifierType: %s" % self.Classifier)
        self.logger.info("Regressor: %s" % self.Regressor)

        self.logger.info("RandomForestTree: MaxFeature : %s, NumEstimators: %d, MinSamplesLeaf: %d, NumJobs: %d, MaxDepth: %d" %
                (self.RFTMaxFeatures, self.RFTNumEstimators, self.RFTMinSamplesLeaf, self.RFTNumJobs, self.RFTMaxDepth))
