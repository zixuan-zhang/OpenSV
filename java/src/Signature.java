/**
 * Created by zixuan on 2016/9/1.
 */

import java.util.ArrayList;
import java.util.HashMap;

public class Signature {
    public Signature()
    {
        _signature = new HashMap<>();
    }

    public Signature(String userName, String signatureID, String sampleID, int label)
    {
        _signature = new HashMap<>();
        this.userName = userName;
        this.signatureID = signatureID;
        this.sampleID = sampleID;
        this.label = label;
    }

    public Signature(String userName, String signatureID, int label)
    {
        _signature = new HashMap<>();
        this.userName = userName;
        this.signatureID = signatureID;
        this.label = label;
        this.sampleID = "";
    }

    public Signature(String userName, int label)
    {
        _signature = new HashMap<>();
        this.userName = userName;
        this.label = label;
    }

    public void SetCom(String com, ArrayList<Double> sequence)
    {
        _signature.put(com, sequence);
    }

    public ArrayList<Double> GetCom(String com)
    {
        if (_signature.containsKey(com))
            return _signature.get(com);
        else
            return null;
    }

    private HashMap<String, ArrayList<Double>> _signature;

    private String userName;

    private String signatureID;

    private String sampleID;

    private int label;
}
