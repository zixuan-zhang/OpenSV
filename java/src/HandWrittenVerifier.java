import cn.ac.iscas.handwriter.MainActivity;
import java.util.ArrayList;

public class HandWrittenVerifier
{
    public boolean Register()
    {
        _records = MainActivity.getrecords();
        if (_records.size() != 5)
        {
            return false;
        }
    }

    public boolean Verify()
    {
        ArrayList records = MainActivity.getrecords();
        return true;
    }

    private ArrayList _records;

    private String ClassifierDumpFilePath;

    private HandWrittenVerifier()
    {
        ClassifierDumpFilePath = "";
    }
};
