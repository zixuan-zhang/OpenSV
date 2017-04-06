// This autogenerated skeleton file illustrates how to build a server.
// You should copy it to another filename to avoid overwriting it.
//

#include <iostream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

#include "../gen-cpp/HandWriter.h"
#include "../gen-cpp/opensv_constants.h"
#include "../gen-cpp/opensv_types.h"

using namespace std;
using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

using boost::shared_ptr;

using namespace  ::opensv;

class HandWriterHandler : virtual public HandWriterIf {
 public:
  HandWriterHandler() {
    // Your initialization goes here
  }

  int32_t ping(const int32_t num) {
      return num + 1;
  }

  void accountRegister(Ret& _return, const Request& request) {
    cout<<"id is "<<request.id<<endl;
    for (int i = 0; i < request.signatures.size(); ++i) {
        Signature sig = request.signatures[i];
        for (int j = 0; j < sig.points.size(); ++j) {
            Point point = sig.points[j];
            cout<<point.t<<" "<<point.x<<" "<<point.y<<" "<<point.p<<endl;
        }
    }
    Ret ret;
    ret.success = true;
    _return = ret;
  }

  void verify(Ret& _return, const Request& request) {
    // Your implementation goes here
    printf("verify\n");
  }

};

int main(int argc, char **argv) {
  int port = 9090;
  shared_ptr<HandWriterHandler> handler(new HandWriterHandler());
  shared_ptr<TProcessor> processor(new HandWriterProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}

