/**
 * Created by zixuan on 2016/9/1.
 */

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Level;

public class Config{
    public static final Double PreWidth = 400.0;
    public static final Double PreHeight = 200.0;

    public static final String Template = "template";
    public static final String Avg = "avg";
    public static final String Min = "min";
    public static final String Max = "max";
    public static final String Median = "median";

    public static final String Forgery = "FORGERY";
    public static final String Genuine = "GENUINE";
    /** SUSIG database settings. */
    public static final String SusigDataPath = "F:\\data\\SUSig";
    public static final String BlindSubCorpus = "BlindSubCorpus";
    public static final String VisualSubCorpus = "VisualSubCorpus";
    /** SVC2004 database settings. */
    public static final String SvcDataPath = "F:\\data\\Task2";
    /** Self database settings. */
    public static final String SelfDataPath = "F:\\data\\self";

    public static final String SelfDatabase = "Self";
    public static final String SUSIGDatabase = "SUSIG";
    public static final String SvcDatabase = "SVC";

    public static final String Database = SelfDatabase;

    public static final Double TrainSetRate = 0.7;

    public static final String LogPath = "F:\\data\\log\\OpenSV.log";

    public static ArrayList<String> SigComList = new ArrayList<String>(Arrays.asList("X", "Y", "VX", "VY"));

    public static int DTWMethod= 1;

    public static final int RefCount = 5;

    public static final String DumpFilePath = "F:\\data\\classifer.dump";

    public static HashMap<String, Double> Penalization = new HashMap<String, Double>(){
        {
            put("X", 7.0);
            put("Y", 5.0);
            put("VX", 3.0);
            put("VY", 2.0);
            put("P", 2.0);
            put("VP", 2.0);
        }
    };

    public static HashMap<String, Double> Threshold = new HashMap<String, Double>(){
        {
            put("X", 8.0);
            put("Y", 6.0);
            put("VX", 0.0);
            put("VY", 1.0);
            put("P", 2.0);
            put("VP", 0.0);
        }
    };

    private static ArrayList<String> getFeatureType()
    {
        return new ArrayList<String>(Arrays.asList(Config.Min, Config.Median, Config.Template));
    }

    public static HashMap<String, ArrayList<String>> FeatureType = new HashMap<String, ArrayList<String>>(){
        {
            put("X", getFeatureType());
            put("Y", getFeatureType());
            put("VX", getFeatureType());
            put("VY", getFeatureType());
            put("P", getFeatureType());
            put("VP", getFeatureType());
        }
    };

    public static void ShowConfig()
    {
        // Utils.logger.log(Level.INFO, "ConfigSet : DataSet : {0}", Database);
        Utils.logger.log(Level.INFO, "ConfigSet : TrainSetRate {0} : SigComList {1} : DTWMethod {2} : RefCount {3} : Penalization {4} : Threshold {5} : FeatureType {6}",
                new Object[]{TrainSetRate, SigComList, DTWMethod, RefCount, Penalization, Threshold, FeatureType});
        /*
        Utils.logger.log(Level.INFO, "ConfigSet : TrainSetRate : {0}", TrainSetRate);
        Utils.logger.log(Level.INFO, "ConfigSet : SigComList : {0}", SigComList);
        Utils.logger.log(Level.INFO, "ConfigSet : DTWMethod : {0}", DTWMethod);
        Utils.logger.log(Level.INFO, "ConfigSet : RefCount : {0}", RefCount);
        Utils.logger.log(Level.INFO, "ConfigSet : Penalization : {0}", Penalization);
        Utils.logger.log(Level.INFO, "ConfigSet : Threshold : {0}", Threshold);
        Utils.logger.log(Level.INFO, "ConfigSet : FeatureType : {0}", FeatureType);
        */
    }
}
