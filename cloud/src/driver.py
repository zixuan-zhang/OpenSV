#coding=utf-8

import os
import cPickle

import settings
import processor
from person import PersonTest
from database import SignatureStorage
from opensv.ttypes import Ret, ErrorCode

class HandWriterDriver(object):
    def __init__(self, config):
        self.config = config
        self.database = SignatureStorage(config)
        self.model = self._load_model()
        self.processor = processor.PreProcessor()

    def accountRegister(self, account, signatures):
        # Pre process signature and then reconsturct them
        for sig in signatures:
            self._pre_process_for_signle_signature(sig)
            self._reconstruct_signature(sig)
        self.database.save(account, signatures)
        return Ret(True, None)

    def verify(self, account, test_signature):
        # TODO: get reference signatures from db.

        reference_signatures = self.database.load(account)
        if not reference_signatures:
            return Ret(False, ErrorCode.ReferenceSignatureShortage)
        print reference_signatures
        print "reference signature count : %d" % len(reference_signatures)
        test_signature = self._pre_process_for_signle_signature(test_signature)
        test_signature = self._reconstruct_signature(test_signature)
        personTest = PersonTest(self.config, reference_signatures)
        features = personTest.calc_dis(test_signature)
        res = self.model.predict(features)
        return Ret(res, None)

    def _load_model(self):
        # TODO: load trained model.
        model = None
        with open(self.config.ModelDumpFilePath) as fp:
            model = cPickle.load(fp)
        return model

    def _pre_process_for_signle_signature(self, signature):
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

    def _reconstruct_signature(self, signature):
        # Reconstruct signature to dictionary like object
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
