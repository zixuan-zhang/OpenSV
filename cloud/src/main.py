# encoding: utf-8

##############################################################################
####   File         : main.py
####   Author       : zixuan.zhang.victor@gmail.com
####   Create Date  : 2017年03月16日 星期四 22时38分28秒
##############################################################################

import glob
import sys
import logging
import time

logging.basicConfig()

sys.path.insert(0, "gen-py")
# sys.path.insert(0, glob.glob("../../lib/py/build/lib*")[0])

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from opensv import HandWriter
from opensv.ttypes import Point, Signature, Request, Ret, ErrorCode

from config import Config
from driver import HandWriterDriver


class HandWriterHandler:
    def __init__(self):
        config = Config()
        self.driver = HandWriterDriver(config)
    
    def ping(self, num):
        print "Receive ping() request, num is %d" % num
        return num + 1

    def accountRegister(self, request):
        print "Receive accountRegister() request"
        account_id = request.id
        signatures = self._extract_signatures(request)
        if len(signatures) < 5:
            return Ret(False, ErrorCode.ReferenceSignatureShortage)

        # TODO: futher check of signature quality

        # Forward driver to process
        return self.driver.accountRegister(account_id, signatures)

    def verify(self, request):
        print "Receive verify() request"
        account_id = request.id
        signatures = self._extract_signatures(request)
        if len(signatures) < 1:
            return Ret(False, ErrorCode.TestSignatureNotFound)
        signature = signatures[0]
        return self.driver.verify(account_id, signature)

    def _extract_signatures(self, request):
        signatures = []
        # TODO: extract signature from request.

        for sig in request.signatures:
            t_list = []
            x_list = []
            y_list = []
            p_list = []
            for point in sig.points:
                t_list.append(point.t)
                x_list.append(point.x)
                y_list.append(point.y)
                p_list.append(point.p)
            signature = {
                    "T": t_list,
                    "X": x_list,
                    "Y": y_list,
                    "P": p_list}
            signatures.append(signature)
        return signatures

if __name__ == "__main__":
    handler = HandWriterHandler()
    processor = HandWriter.Processor(handler)
    transport = TSocket.TServerSocket(port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    print "Starting the server..."
    server.serve()
    print "Done."
