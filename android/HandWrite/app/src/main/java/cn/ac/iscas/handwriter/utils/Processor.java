package cn.ac.iscas.handwriter.utils;

/**
 * Created by zixuan on 2016/11/1.
 */

/**
 * Created by zixuan on 2016/9/1.
 */

import java.lang.reflect.Array;
import java.util.ArrayList;

public class Processor {
    public static void size_normalization(ArrayList<Double> X, ArrayList<Double> Y, Double width, Double height)
    {
        Double minX = Utils.GetMinValue(X);
        Double maxX = Utils.GetMaxValue(X);
        Double minY = Utils.GetMinValue(Y);
        Double maxY = Utils.GetMaxValue(Y);

        Double rangeX = maxX - minX;
        Double rangeY = maxY - minY;

        for (int i = 0; i < X.size(); ++i)
            X.set(i, width * (X.get(i) - minX) / rangeX);
        for (int i = 0; i < Y.size(); ++i)
            Y.set(i, height * (Y.get(i) - minY) / rangeY);
    }

    public static void location_normalization(ArrayList<Double> X, ArrayList<Double> Y)
    {
        Double meanX = Utils.GetMeanValue(X);
        Double meanY = Utils.GetMeanValue(Y);
        for (int i = 0; i < X.size(); ++i)
            X.set(i, X.get(i) - meanX);
        for (int i = 0; i < Y.size(); ++i)
            Y.set(i, Y.get(i) - meanY);
    }

    public static ArrayList<Double> calculate_delta(ArrayList<Double> N, ArrayList<Double> T)
    {
        assert(N.size() == T.size());
        ArrayList<Double> newList = new ArrayList<>();
        for (int i = 0; i < N.size() - 1; ++i) {
            if (!T.get(i + 1).equals(T.get(i)))
                newList.add((N.get(i + 1) - N.get(i)) / (T.get(i + 1) - T.get(i)));
            else
                newList.add(0.0);
        }
        return newList;
    }
}


