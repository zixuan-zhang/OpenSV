/*******************************************************************************
 *  @File  : client.cpp
 *  @Author: Zhang Zixuan
 *  @Email : zixuan.zhang.victor@gmail.com
 *  @Blog  : www.noathinker.com
 *  @Date  : 2017年04月06日 星期四 21时26分40秒
 ******************************************************************************/

#include <iostream>
#include <vector>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "../gen-cpp/HandWriter.h"


using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace opensv;


int main() {
    boost::shared_ptr<TTransport> socket(new TSocket("localhost", 9090));
    boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));

    HandWriterClient client(protocol);
    try {
        transport->open();

        Request request;
        request.id = 123;
        vector<Signature> signatures;
        for (int i = 0; i < 5; ++i) {
            Signature signature;
            vector<Point> points;
            for (int j = 0; j < 4; ++j) {
                Point point;
                point.t = i * j * 1;
                point.x = i * j * 15 + 71;
                point.y = i * j * 32 + 71;
                point.p = i * j * 10 + 71;
                points.push_back(point);
            }
            signature.points = points;
            signatures.push_back(signature);
        }
        request.signatures = signatures;
        Ret ret;
        client.accountRegister(ret, request);
        cout<<ret.success<<endl;
        cout<<ret.error<<endl;

        Request verifyRequest;
        request.id = 123;
        vector<Signature> verifySignatures;
        for (int i = 0; i < 1; ++i) {
            Signature signature;
            vector<Point> points;
            for (int i = 0; i < 4; ++i) {
                Point point;
                point.t = i;
                point.x = i * 20 + 17;
                point.y = i * 15 + 17;
                point.p = i * 32 + 17;
                points.push_back(point);
            }
            signature.points = points;
            verifySignatures.push_back(signature);
        }
        request.signatures = signatures;
        Ret verifyRet;
        client.verify(verifyRet, request);
        cout<<verifyRet.success<<endl;
        cout<<verifyRet.error<<endl;

    } catch (TException& tx) {
        cout<<"Error: "<<tx.what()<<endl;
    }


    return 0;
}
