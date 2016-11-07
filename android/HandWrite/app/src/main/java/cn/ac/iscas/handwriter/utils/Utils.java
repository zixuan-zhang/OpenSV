package cn.ac.iscas.handwriter.utils;

/**
 * Created by zixuan on 2016/11/1.
 */

/**
 * Created by zixuan on 2016/9/1.
 */
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.List;
import java.util.logging.SimpleFormatter;

public class Utils {
    public static final Logger logger = Logger.getLogger("OpenSV");

    public static Double GetMeanValue(List<Double> nums)
    {
        Double sum = 0.0;
        for (Double num : nums)
            sum += num;
        return sum / nums.size();
    }

    public static Double GetMedianValue(List<Double> nums)
    {
        assert(nums.size() > 1);
        List<Double> newNums = new ArrayList<Double>();
        newNums.addAll(nums);
        Collections.sort(newNums);
        return ((newNums.size() & 0x1) == 1) ? newNums.get(nums.size() / 2) : (newNums.get(nums.size() / 2) + newNums.get(nums.size() / 2 - 1)) / 2;
    }

    public static int GetMinIndex(List<Double> nums)
    {
        Double _min = nums.get(0);
        int index = 0;
        for (int i = 1; i < nums.size(); ++i)
        {
            if (_min > nums.get(i))
            {
                _min = nums.get(i);
                index = i;
            }
        }
        return index;
    }

    public static Double GetMaxValue(List<Double> nums)
    {
        Double _max = nums.get(0);
        for (int i = 1; i < nums.size(); ++i)
        {
            _max = Math.max(_max, nums.get(i));
        }
        return _max;
    }

    public static Double GetMinValue(List<Double> nums)
    {
        Double _min = nums.get(0);
        for (int i = 1; i < nums.size(); ++i)
            _min = Math.min(_min, nums.get(i));
        return _min;
    }

    public static Boolean Init()
    {
        try
        {
            Date date = new Date();
            SimpleDateFormat df = new SimpleDateFormat("MMddHHmmss");
            String postFix = df.format(date);
            FileHandler fh = new FileHandler(Config.LogPath + "." + postFix);
            Utils.logger.addHandler(fh);
            SimpleFormatter formatter = new SimpleFormatter();
            fh.setFormatter(formatter);
            return Boolean.TRUE;
        }
        catch (IOException e)
        {
            Utils.logger.log(Level.INFO, "Error: when create file handler {0}", e.toString());
            return Boolean.FALSE;
        }
    }

    /*
    public Boolean Final()
    {
        Utils.logger.removeHandler(fh);
        return Boolean.TRUE;
    }
    */
}


