#coding:utf-8

import numpy

class Person(object):
    """
        Basic class of Person.
    """
    def __init__(self, config, refSigs, testSigs):
        """
            @param : config : configuration object
            @param : refSigs : reference signatures
            @param : testSigs : test signatures
        """

        self.config = config
        self.refSigs = refSigs
        self.testSigs = testSigs
        self.refCount = len(refSigs)
        self.templateSig = None
        # select template signature
        self.select_template()
        # calculate base distance
        self.calc_base_dis()

    def select_template(self):
        """
            Just select template signature.
            TODO: different signal with variant weight
        """
        def _size_norm(values):
            minValue = min(values)
            maxValue = max(values)
            _range = maxValue - minValue
            mV = 100.
            values = [mV * (v - minValue) / _range for v in values]
            return values

        self.config.logger.info("selecting template signature")
        refDis = []
        for i in range(self.refCount):
            dis = 0.0
            for j in range(self.refCount):
                if i == j:
                    continue
                comDisList = []
                totalWeight = 0.0
                for com in self.config.SigCompList:
                    signal1 = self.refSigs[i][com]
                    signal2 = self.refSigs[j][com]
                    signal1 = _size_norm(signal1)
                    signal2 = _size_norm(signal2)
                    comDisList.append(self.naive_dtw(signal1, signal2,
                        self.config.Penalization[com], self.config.Threshold[com]) * \
                                self.config.SignalWeight[com])
                    totalWeight += self.config.SignalWeight[com]
                dis += (sum(comDisList) / totalWeight)
            refDis.append(dis)

        self.templateIndex = refDis.index(min(refDis))
        self.config.logger.debug("template index : %d. RefSigDisList: %s" % (self.templateIndex, refDis))
        self.templateSig = self.refSigs[self.templateIndex]

    def calc_base_dis(self):
        """
        Calculate the base value in signal component of signatures
        """
        self.config.logger.info("Calculating base distance")
        self.base = {}

        for com in self.config.SigCompList:
            templateComList = []
            maxComList = []
            minComList = []
            avgComList = []
            medComList = []
            for i in range(self.refCount):
                if i == self.templateIndex:
                    continue
                comi = self.refSigs[i][com]
                templateComDis = self.naive_dtw(comi, self.templateSig[com],
                        self.config.Penalization[com], self.config.Threshold[com])
                templateComList.append(templateComDis)
                comDisList = []
                for j in range(self.refCount):
                    if i == j:
                        continue
                    comj = self.refSigs[j][com]
                    comDisList.append(self.naive_dtw(comi, comj,
                        self.config.Penalization[com], self.config.Threshold[com]))
                maxComList.append(max(comDisList))
                minComList.append(min(comDisList))
                avgComList.append(numpy.mean(comDisList))
                medComList.append(numpy.median(comDisList))
            if "template" in self.config.FeatureType[com]:
                self.base["template" + com] = numpy.mean(templateComList)
            if "max" in self.config.FeatureType[com]:
                self.base["max"+com] = numpy.mean(maxComList)
            if "min" in self.config.FeatureType[com]:
                self.base["min"+com] = numpy.mean(minComList)
            if "avg" in self.config.FeatureType[com]:
                self.base["avg"+com] = numpy.mean(avgComList)
            if "med" in self.config.FeatureType[com]:
                self.base["med"+com] = numpy.mean(medComList)
            self.config.logger.debug("Calculating signal: %s. %s" % (com,
                ", ".join(["%s:%s"%(items[0], items[1]) for items in self.base.items()])))

    def calc_dis(self, signature):
        """
            For given signature, calculate vector[] with normalization
        """
        featureVec = []
        for com in self.config.SigCompList:
            comSig = signature[com]
            comTem = self.templateSig[com]
            templateComDis = self.naive_dtw(comSig, comTem,
                    self.config.Penalization[com], self.config.Threshold[com])
            comDisList = []
            for i in range(self.refCount):
                comI = self.refSigs[i][com]
                dis = self.naive_dtw(comSig, comI,
                        self.config.Penalization[com], self.config.Threshold[com])
                comDisList.append(dis)
            maxComDis = max(comDisList)
            minComDis = min(comDisList)
            avgComDis = numpy.mean(comDisList)
            medComDis = numpy.median(comDisList)
            if "template" in self.config.FeatureType[com]:
                featureVec.append(templateComDis / self.base["template"+com])
            if "max" in self.config.FeatureType[com]:
                featureVec.append(maxComDis / self.base["max"+com])
            if "min" in self.config.FeatureType[com]:
                featureVec.append(minComDis / self.base["min"+com])
            if "avg" in self.config.FeatureType[com]:
                featureVec.append(avgComDis / self.base["avg"+com])
            if "med" in self.config.FeatureType[com]:
                featureVec.append(medComDis / self.base["med"+com])
            
        return featureVec

    def naive_dtw(self, A, B, p=5, t=5):
        penalization = p
        threshold = t
        len1 = len(A)
        len2 = len(B)
        distance = numpy.zeros((len1, len2))
        # initialize distance values
        distance[0][0] = abs(A[0] - B[0])
        for i in range(1, len1):
            distance[i][0] = distance[i-1][0] + abs(A[i] - B[0])

        for j in range(1, len2):
            distance[0][j] = distance[0][j-1] + abs(A[0] - B[j])

        for i in range(1, len1):
            for j in range(1, len2):
                if self.config.DTWMethod == 1:
                    distance[i][j] = min([distance[i-1][j], distance[i][j-1],
                            distance[i-1][j-1]]) + abs(A[i]-B[j])
                elif self.config.DTWMethod == 2:
                    # DTWMethod 2
                    d1 = distance[i-1][j] + penalization
                    d2 = distance[i][j-1] + penalization
                    other = 0 if (abs(A[i] - B[j]) < threshold) else (abs(A[i] - B[j]) - threshold)
                    d3 = distance[i-1][j-1] + other
                    distance[i][j] = min([d1, d2, d3])
                elif self.config.DTWMethod == 3:
                    d1 = distance[i-1][j] + abs(A[i] - B[j])
                    d2 = distance[i][j-1] + abs(A[i] - B[j])
                    other = 0 if (abs(A[i] - B[j]) < threshold) else (abs(A[i] - B[j]) - threshold)
                    d3 = distance[i-1][j-1] + other
                    distance[i][j] = min([d1, d2, d3])
                    
        return distance[len1-1][len2-1]

class PersonTest(Person):
    def __init__(self, config, refSigs):
        super(PersonTest, self).__init__(config, refSigs, None)
