# encoding: utf-8

"""
@File    : handwrite_unlocker.py
@File    : zixuan.zhang.victor@gmail.com
@Create Date : Wed 14 Dec 2016 03:42:45 AM EST
"""

import os
from sklearn.externals import joblib

import processor
import self_config
import person

class Handwrite(object):
    """
    Class to
    """

    def __init__(self, config):
        """
        Parameters
        -----------
        config : DatasetConfig object. contain basic configurations that needed.
        """
        modelPath = "model.pkl"
        if not os.path.exists(modelPath):
            self.model = joblib.load(modelPath)
        self.config = config
        self.model = None
        self.processor = processor.PreProcessor()

    def predict(self, signatureFile):
        """
        This function is to predict if the signature given is valid.

        Parameters
        ------------
        @signatureFile : string : file path that contain the signature

        Returns
        ------------
        True for genuine signature else False
        """

        if not self.model:
            raise Exception("ModelNotSet error")

        # Extract signature object
        signature = {"T": [], "X": [], "Y": [], "P": []}
        with open(signatureFile) as fp:
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
            signature = self._pre_process_for_signle_signature(signature)
            signature = self._reconstruct_signature(signature)

        # TODO: get reference signature
        refSigs = None

        # Calculate feature
        personTest = person.PersonTest(self.config, refSigs)
        dis = personTest.calc_dis(signature)

        result = self.model.predict(dis)
        return result

    def _pre_process_for_signle_signature(self, signature):
        """
        Internal function. Do not use this from outer space.
        This function will do pre-processing for given signature
        according to configuration.
        Basicly two steps are followed: 
            1. size normalization
            2. location normalization or offset to origin normalization
        """
        RX = signature["X"]
        RY = signature["Y"]
        if self.config.PreProcessorSwitch:
            if self.config.SizeNormSwitch:
                RX, RY = self.processor.size_normalization(RX, RY, 400, 200)
            if self.config.LocalNormalType == "mid":
                RX, RY = self.processor.location_normalization(RX, RY)
            elif self.config.LocalNormalType == "offset":
                RX, RY = self.processor.offset_to_origin_normalization(RX, RY)
            signature["X"] = RX
            signature["Y"] = RY
        return signature

    def _reconstruct_signature(self, signature):
        """
        This function suppose that the original signature contains only
        T, X, Y and P signals, and this function will generate VX and VY.
        
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

if __name__ == "__main__":
    config = self_config.SelfConfig()
    handWrite = Handwrite(config)
