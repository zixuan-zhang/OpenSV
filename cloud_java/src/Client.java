/**
 * Created by zixuan on 2017/3/21.
 */


import org.apache.thrift.TException;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TSSLTransportFactory.TSSLTransportParameters;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;

import java.util.ArrayList;
import java.util.List;
import java.util.ArrayList;

public class Client {
    public static void main(String[] args) {
        try {
            TTransport transport = new TSocket("localhost", 9090);
            transport.open();
            TProtocol protocol = new TBinaryProtocol(transport);
            HandWriter.Client client = new HandWriter.Client(protocol);
            perform(client);
            transport.close();
        } catch (TException x) {
            x.printStackTrace();
        }
    }

    private static void perform(HandWriter.Client client) throws TException {
        int res = client.ping(456);
        System.out.println("456 ping res is " + res);

        List<Signature> signatures = new ArrayList<Signature>();
        for (int i = 0; i < 5; ++i) {
            List<Point> points = new ArrayList<Point>();
            for (int j = 0; j < 5; ++j) {
                Point point = new Point(100*j, 100*j, 100*j, 100*j);
                points.add(point);
            }
            Signature signature = new Signature(points);
            signatures.add(signature);
        }

        Request request = new Request(123456, signatures);

        Ret ret = client.accountRegister(request);
        System.out.println("Registering...");
        System.out.println("Result " + ret.isSuccess() + ", error code is " + ret.getError());

    }
}
