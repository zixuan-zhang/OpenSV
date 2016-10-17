import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by zixuan on 2016/9/3.
 */
public class PersonSignatures {
    public PersonSignatures(String userName, ArrayList<Signature> genuines, ArrayList<Signature> forgeries)
    {
        this.userName = userName;
        this._signatures = new HashMap<>();
        this._signatures.put("genuine", genuines);
        this._signatures.put("forgery", forgeries);
    }

    public PersonSignatures(String userName)
    {
        this.userName = userName;
        this._signatures = new HashMap<>();
    }

    public ArrayList<Signature> GetGenuine()
    {
        return this._signatures.get("genuine");
    }

    public ArrayList<Signature> GetForgery()
    {
        return this._signatures.get("forgery");
    }

    public String GetUserName()
    {
        return userName;
    }

    private HashMap<String, ArrayList<Signature>> _signatures;

    private String userName;
}
