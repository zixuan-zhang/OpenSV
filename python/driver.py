#coding=utf-8

import os
import cPickle

import settings
import processor
from person import PersonTraining, PersonTest

class BaseDriver(object):
    def __init__(self, config):
        self.config = config
        self.driver = self.config.model
        self.processor = processor.PreProcessor()
        if self.config.Dataset == settings.SUSIG:
            trainSets, testSets = self.get_signatures_from_susig()
        elif self.config.Dataset == settings.Self:
            trainSets, testSets = self.get_signatures_from_self()
        self.train_set = trainSets
        self.test_set = testSets

        self.config.logger.info("Train set count : %d" % len(trainSets))
        self.config.logger.info("Test set count : %d" % len(testSets))
        self.config.logger.info("Genuine count in train set : %d" % len(trainSets[0]["genuine"]))
        self.config.logger.info("Forgery count in train set : %d" % len(trainSets[0]["forgery"]))
        self.config.logger.info("Genuine count in test set : %d" % len(testSets[0]["genuine"]))
        self.config.logger.info("Forgery count in test set : %d" % len(testSets[0]["forgery"]))

    def pre_process_for_signle_signature(self, signature):
        T  = signature["T"]
        RX = signature["X"]
        RY = signature["Y"]
        if self.config.PreProcessorSwitch:
            if self.config.SizeNormSwitch:
                RX, RY = self.processor.size_normalization(RX, RY, 400, 200)
            if self.config.LocalNormalType== "mid":
                RX, RY = self.processor.location_normalization(RX, RY)
            elif self.config.LocalNormalType== "offset":
                RX, RY = self.processor.offset_to_origin_normalization(RX, RY)
            signature["X"] = RX
            signature["Y"] = RY
        return signature

    def reconstructSignature(self, signature):
        """
        Reconstruct signature to dictionary like object
        """
        def _calculate_delta(T, valueList):
            """
            This function just calculate the derivatives of valueList.
            """
            assert len(T) == len(valueList)
            newValueList = []
            for i in range(1, len(T)):
                slot = T[i] - T[i-1]
                value = (valueList[i] - valueList[i-1]) / slot \
                        if slot != 0 else None
                if value:
                    newValueList.append(value)
            return newValueList

        T = signature["T"]
        X = signature["X"]
        Y = signature["Y"]
        VX = _calculate_delta(T, X)
        VY = _calculate_delta(T, Y)
        signature["VX"] = VX
        signature["VY"] = VY
        return signature

    def get_signatures_from_susig(self):
        """
        Load original data from susig
        @return: trainSets, testSets
        """
        def _get_signatures_from_susig_folder(folder):
            
            signatures = {}
            for fileName in os.listdir(folder):
                filePath = os.path.join(folder, fileName)
                signature = {"T": [], "X": [], "Y": [], "P": []}
                with open(filePath) as fp:
                    lines = fp.readlines()
                    for line in lines[2:]:
                        line = line.strip()
                        if not line:
                            continue
                        items = line.split()
                        signature["X"].append(float(items[0]))
                        signature["Y"].append(float(items[1]))
                        signature["T"].append(float(items[2]))
                        signature["P"].append(float(items[3]))
                    personID = fileName.split(".")[0].split("_")[0]
                    if personID not in signatures:
                        signatures[personID] = []
                    signature = self.pre_process_for_signle_signature(signature)
                    signature = self.reconstructSignature(signature)
                    signatures[personID].append(signature)

            return signatures
        self.config.logger.info("Getting signatures from susig")
        trainSignatures = {}

        # For train forgery
        trainForgeryFolder = "../data/SUSig/VisualSubCorpus/FORGERY"
        trainForgerySubSigs = _get_signatures_from_susig_folder(trainForgeryFolder)
        for (personID, value) in trainForgerySubSigs.items():
            if personID not in trainSignatures:
                trainSignatures[personID] = {"genuine": [], "forgery": []}
            trainSignatures[personID]["forgery"].extend(value)

        # For train genuine session1
        trainGenuineSession1Folder = "../data/SUSig/VisualSubCorpus/GENUINE/SESSION1"
        trainGenuineSession1SubSigs= _get_signatures_from_susig_folder(trainGenuineSession1Folder)
        for (personID, value) in trainGenuineSession1SubSigs.items():
            if personID not in trainSignatures:
                trainSignatures[personID] = {"genuine": [], "forgery": []}
            trainSignatures[personID]["genuine"].extend(value)

        if self.config.MultiSession:
            # For train genuine session2
            trainGenuineSession2Folder = "../data/SUSig/VisualSubCorpus/GENUINE/SESSION2"
            trainGenuineSession2SubSigs = _get_signatures_from_susig_folder(trainGenuineSession2Folder)
            for (personID, value) in trainGenuineSession2SubSigs.items():
                if personID not in trainSignatures:
                    trainSignatures[personID] = {"genuine": [], "forgery": []}
                trainSignatures[personID]["genuine"].extend(value)

        trainSets = trainSignatures.values()

        # For test sets
        testSignatures = {}
        testForgeryFolder = "../data/SUSig/VisualSubCorpus/VALIDATION/VALIDATION_FORGERY"
        testForgerySubSigs = _get_signatures_from_susig_folder(testForgeryFolder)
        for (personID, value) in testForgerySubSigs.items():
            if personID not in testSignatures:
                testSignatures[personID] = {"genuine": [], "forgery": []}
            testSignatures[personID]["forgery"].extend(value)

        testGenuineFolder = "../data/SUSig/VisualSubCorpus/VALIDATION/VALIDATION_GENUINE"
        testGenuineSubSigs = _get_signatures_from_susig_folder(testGenuineFolder)
        for (personID, value) in testGenuineSubSigs.items():
            if personID not in testSignatures:
                testSignatures[personID] = {"genuine": [], "forgery": []}
            testSignatures[personID]["genuine"].extend(value)
        testSets = testSignatures.values()

        if self.config.Mode== settings.ModeTest:
            return testSets[:1], trainSets[:1]
        else:
            return testSets+trainSets[0:self.config.TrainSetCount-10], trainSets[self.config.TrainSetCount-10:]

    def get_signatures_from_self(self):
        """
            Load original data from self
            @return: trainSets, testSets
        """
        def _get_signatures_from_self_folder(folder):
            signatures = {}
            for fileName in os.listdir(folder):
                filePath = os.path.join(folder, fileName)
                signature = {"T": [], "X": [], "Y": [], "P": []}
                with open(filePath) as fp:
                    lines = fp.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        items = line.split()
                        signature["T"].append(float(items[0]))
                        signature["X"].append(float(items[1]))
                        signature["Y"].append(float(items[2]))
                        signature["P"].append(float(items[3]))
                    maxY = max(signature["Y"])
                    signature["Y"] = [maxY - y for y in signature["Y"]]
                    personID = fileName.split("_")[0]
                    if personID not in signatures:
                        signatures[personID] = []
                    signature = self.pre_process_for_signle_signature(signature)
                    signature = self.reconstructSignature(signature)
                    signatures[personID].append(signature)
            return signatures

        self.config.logger.info("Getting signatures from self")
        # For genuine signatures
        genuineSignatures = _get_signatures_from_self_folder(self.config.GenuineFolder)
        forgerySignatures = _get_signatures_from_self_folder(self.config.ForgeryFolder)
        
        totalSignatures = {}
        for (personID, value) in genuineSignatures.items():
            if personID not in totalSignatures:
                totalSignatures[personID] = {"genuine": [], "forgery": []}
            totalSignatures[personID]["genuine"].extend(value)
        for (personID, value) in forgerySignatures.items():
            if personID not in totalSignatures:
                totalSignatures[personID] = {"genuine": [], "forgery": []}
            totalSignatures[personID]["forgery"].extend(value)
        totalSet = totalSignatures.values()
        trainCount = int(len(totalSet) * self.config.TrainCountRate)
        if self.config.Mode == settings.ModeTest:
            return totalSet[:1], totalSet[1:2]
        else:
            return totalSet[:trainCount], totalSet[trainCount:]

    def test(self):
        pass

class ClassifyDriver(BaseDriver):
    def __init__(self, config):
        super(ClassifyDriver, self).__init__(config)

        genuineX = []
        forgeryX = []

        genuineY = []
        forgeryY = []

        # Training process
        for sigs in self.train_set:
            personTrain = PersonTraining(self.config, sigs)
            genuine, forgery = personTrain.calc_train_set()
            genuineX.extend(genuine)
            forgeryX.extend(forgery)

        genuineY = [1] * len(genuineX)
        forgeryY = [0] * len(forgeryX)

        trainX = genuineX + forgeryX
        trainY = genuineY + forgeryY

        self.driver.fit(trainX, trainY)
        with open(self.config.ModelDumpFilePath, "w") as fp:
            cPickle.dump(self.driver, fp)


    def test(self):
        self.config.logger.info("Start test")
        count = 1
        test_set = self.test_set
        if self.config.TrainSetInclude:
            test_set.extend(self.train_set)
        forgery_test_result = []
        genuine_test_result = []
        random_test_result = []

        genuine_test_dis = []
        forgery_test_dis = []

        for i in range(len(test_set)):
            one_test_set = test_set[i]
            self.config.logger.info("Test signature: %d" % count)
            count += 1
            personTest = PersonTest(self.config, one_test_set["genuine"][0:self.config.RefCount])
            genuine_set = one_test_set["genuine"][self.config.RefCount:]
            forgery_set = one_test_set["forgery"]
            random_set = []

            for j in range(len(genuine_set)):
                sig = genuine_set[j]
                dis = personTest.calc_dis(sig)
                res = self.driver.predict(dis)
                self.config.logger.info("Genuine Test: Result: %s, %s" % (res, dis))
                genuine_test_result.append(res)
                if (res != 1):
                    self.config.logger.fatal("FalseReject: uid: %d, sid: %d" % (i, j))

            for j in range(len(forgery_set)):
                sig = forgery_set[j]
                dis = personTest.calc_dis(sig)
                res = self.driver.predict(dis)
                self.config.logger.info("Forgery Test: Result: %s, %s" % (res, dis))
                forgery_test_result.append(res)
                if (res != 0):
                    self.config.logger.fatal("FalseAccept: uid: %d, sid: %d" % (i, j))

            if self.config.RandomForgeryInclude:
                for j in range(len(test_set)):
                    if i == j:
                        continue
                    random_set.extend(test_set[j]["genuine"])
                    random_set.extend(test_set[j]["forgery"])

                # train set included
                for one_train_set in self.train_set:
                    random_set.extend(one_train_set["genuine"])
                    random_set.extend(one_train_set["forgery"])

                for j in range(len(random_set)):
                    sig = random_set[j]
                    dis = personTest.calc_dis(sig)
                    res = self.driver.predict(dis)
                    self.config.logger.info("Random Test: Result: %s, %s" % (res, dis))
                    random_test_result.append(res)
                    if (res != 0):
                        self.config.logger.fatal("FalseAccept: uid: %d, sig: %d" % (i, j))

        self.config.logger.info("genuine test set count: %d" % len(genuine_test_result))
        self.config.logger.info("true accepted count: %d" % sum(genuine_test_result))
        self.config.logger.info("false rejected rate: %f" % (sum(genuine_test_result) / float(len(genuine_test_result))))

        self.config.logger.info("forgery test set count: %d" % len(forgery_test_result))
        self.config.logger.info("false accepted count: %d" % sum(forgery_test_result))
        self.config.logger.info("false accepted rate: %f" % (1 - sum(forgery_test_result) / float(len(forgery_test_result))))

        if self.config.RandomForgeryInclude:
            self.config.logger.info("random test set count: %d" % len(random_test_result))
            self.config.logger.info("false accepted count: %d" % sum(random_test_result))
            self.config.logger.info("false accepted rate: %f" % (1 - sum(random_test_result) / float(len(random_test_result))))


class RegressionDriver(BaseDriver):
    def __init__(self, config):
        super(RegressionDriver, self).__init__(config)
        genuineX = []
        forgeryX = []

        genuineY = []
        forgeryY = []

        # Training process
        for sigs in self.train_set:
            personTrain = PersonTraining(self.config, sigs)
            genuine, forgery = personTrain.calc_train_set()
            genuineX.extend(genuine)
            forgeryX.extend(forgery)

        # To adjust PCA result, 0 means genuine and 1 means forgery
        genuineY = [0.0] * len(genuineX)
        forgeryY = [1.0] * len(forgeryX)

        trainX = genuineX + forgeryX
        trainY = genuineY + forgeryY

        self.driver.fit(trainX, trainY)

    def test(self):
        self.config.logger.info("Start test")
        count = 1
        test_set = self.test_set
        if self.config.TrainSetInclude:
            test_set.extend(self.train_set)
        forgery_test_result = []
        genuine_test_result = []
        random_test_result = []

        genuine_test_dis = []
        forgery_test_dis = []

        falseRejectCount = 0
        falseAcceptSkillCount = 0
        falseAcceptRandomCount = 0

        for i in range(len(test_set)):
            one_test_set = test_set[i]
            self.config.logger.info("Test signature: %d" % count)
            count += 1
            personTest = PersonTest(self.config, one_test_set["genuine"][0:self.config.RefCount])
            genuine_set = one_test_set["genuine"][self.config.RefCount:]
            forgery_set = one_test_set["forgery"]
            random_set = []

            for j in range(len(genuine_set)):
                sig = genuine_set[j]
                dis = personTest.calc_dis(sig)
                if self.config.Regressor == "PCA":
                    res = self.driver.transform(dis)
                    res = res.tolist()[0][0]
                else:
                    res = self.driver.predict(dis)
                    res = res.tolist()[0]
                genuine_test_dis.append(res)
                self.config.logger.debug("Genuine Test: Result: %s, %s" % (res, dis))
                genuine_test_result.append(res)
                if (res > 0.5):
                    self.config.logger.debug("FalseReject: uid: %d, sid: %d" % (i, j))
                    falseRejectCount += 1

            for j in range(len(forgery_set)):
                sig = forgery_set[j]
                dis = personTest.calc_dis(sig)
                if self.config.Regressor == "PCA":
                    res = self.driver.transform(dis)
                    res = res.tolist()[0][0]
                else:
                    res = self.driver.predict(dis)
                    res = res.tolist()[0]
                forgery_test_dis.append(res)
                self.config.logger.debug("Forgery Test: Result: %s, %s" % (res, dis))
                forgery_test_result.append(res)
                if (res <= 0.5):
                    self.config.logger.debug("FalseAccept: uid: %d, sid: %d" % (i, j))
                    falseAcceptSkillCount += 1

            if self.config.RandomForgeryInclude:
                for j in range(len(test_set)):
                    if i == j:
                        continue
                    random_set.extend(test_set[j]["genuine"])
                    random_set.extend(test_set[j]["forgery"])

                # train set included
                for one_train_set in self.train_set:
                    random_set.extend(one_train_set["genuine"])
                    random_set.extend(one_train_set["forgery"])

                for j in range(len(random_set)):
                    sig = random_set[j]
                    dis = personTest.calc_dis(sig)
                    if self.config.Regressor == "PCA":
                        res = self.driver.transform(dis)
                        res = res.tolist()[0][0]
                    else:
                        res = self.driver.predict(dis)
                        res = res.tolist()[0]
                    forgery_test_dis.append(res)
                    self.config.logger.debug("Random Test: Result: %s, %s" % (res, dis))
                    random_test_result.append(res)
                    if (res <= 0.5):
                        self.config.logger.debug("FalseAccept: uid: %d, sig: %d" % (i, j))
                        falseAcceptRandomCount += 1

        self.config.logger.info("genuine test set count: %d" % len(genuine_test_result))
        self.config.logger.info("false reject count: %d" % falseRejectCount)
        self.config.logger.info("false rejected rate: %f" % (float(falseRejectCount) / float(len(genuine_test_result))))

        self.config.logger.info("forgery test set count: %d" % len(forgery_test_result))
        self.config.logger.info("false accepted count: %d" % falseAcceptSkillCount)
        self.config.logger.info("false accepted rate: %f" % (float(falseAcceptSkillCount) / float(len(forgery_test_result))))

        if self.config.RandomForgeryInclude:
            self.config.logger.info("random test set count: %d" % len(random_test_result))
            self.config.logger.info("false accepted count: %d" % falseAcceptRandomCount)
            self.config.logger.info("false accepted rate: %f" % (float(falseAcceptRandomCount) / float(len(random_test_result))))

        # Compute Equal Error Rate
        genuine_test_dis_list = sorted(genuine_test_dis, reverse=True) # desending order
        forgery_test_dis_list = sorted(forgery_test_dis) # asending order

        lastGap = 100.0
        genuineCount = len(genuine_test_dis_list)
        forgeryCount = len(forgery_test_dis_list)
        falseRejectRate = None
        falseAcceptRate = None
        for i in range(genuineCount):
            pivotal = genuine_test_dis_list[i]
            falseRejectRate = float(i) / genuineCount
            j = 0
            while j < forgeryCount and forgery_test_dis_list[j] <= pivotal:
                j += 1
            falseAcceptRate = float(j) / forgeryCount
            gap = abs(falseAcceptRate - falseRejectRate)
            if gap == 0.0:
                break;
            elif gap < lastGap:
                lastGap = gap
            else:
                break
        self.config.logger.info("falseRejectRate: %f, falseAcceptRate: %f, gap: %f" % (falseRejectRate, falseAcceptRate, lastGap))
        self.config.logger.info("TestResultGenuine : %s" % genuine_test_dis)
        self.config.logger.info("TestResultForgery : %s" % forgery_test_dis)
