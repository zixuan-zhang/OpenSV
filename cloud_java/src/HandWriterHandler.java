/**
 * Created by zixuan on 2017/3/21.
 */

import org.apache.thrift.TException;

import java.util.List;

public class HandWriterHandler implements HandWriter.Iface{
    public HandWriterHandler() {

    }

    public int ping(int num) {
        System.out.println("Receive ping" + num);
        return num + 1;
    }

    public Ret accountRegister(Request request) {
        System.out.println("Receive account register");
        int id = request.getId();
        List<Signature> signatures = request.getSignatures();
        System.out.println("Account id " + id + ", Signature count " + signatures.size());
        return new Ret(true, null);
    }

    public Ret verify(Request request) {
        System.out.println("Receive verify");
        int id = request.getId();
        List<Signature> signatures = request.getSignatures();
        System.out.println("Account id " + id + ", Signature count " + signatures.size());
        return new Ret(true, null);
    }
}
