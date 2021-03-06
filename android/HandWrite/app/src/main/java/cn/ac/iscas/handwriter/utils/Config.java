package cn.ac.iscas.handwriter.utils;

/**
 * Created by zixuan on 2016/9/1.
 */

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;

public class Config{
    public static final Double PreWidth = 400.0;
    public static final Double PreHeight = 200.0;

    public static final String Template = "template";
    public static final String Avg = "avg";
    public static final String Min = "min";
    public static final String Max = "max";
    public static final String Median = "median";

    public static ArrayList<String> SigComList = new ArrayList<String>(Arrays.asList("X", "Y", "VX", "VY", "P"));

    public static int DTWMethod= 1;
    public static final int RefCount = 5;
    public static final String DatabaseTableName = "signature_records";

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

    public static ArrayList<String> GetFeatureType()
    {
        return new ArrayList<String>(Arrays.asList(Config.Min, Config.Template));
    }

    public static HashMap<String, ArrayList<String>> FeatureType = new HashMap<String, ArrayList<String>>(){
        {
            put("X", GetFeatureType());
            put("Y", GetFeatureType());
            put("VX", GetFeatureType());
            put("VY", GetFeatureType());
            put("P", GetFeatureType());
            put("VP", GetFeatureType());
        }
    };

}


