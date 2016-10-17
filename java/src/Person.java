/**
 * Created by zixuan on 2016/9/1.
 */

import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Level;

public class Person {
    public Person(ArrayList<Signature> refSigs)
    {
        // Utils.logger.log(Level.INFO, "Reference signature count: {0}", refSigs.size());
        this.refSigs = refSigs;
        // Select template signature
        this.SelectTemplate();
        // Calculate base distance
        this.baseValueMap = new HashMap<>();
        this.CalcBaseDis();
    }


    /*
    @Description: Select template signature from reference signatures.
     */
    public void SelectTemplate()
    {
        // Utils.logger.log(Level.INFO, "Selecting template signature");
        ArrayList<Double> refDis = new ArrayList<>();
        for (int i = 0; i < this.GetRefCount(); ++i)
        {
            double dis = 0.0;
            for (int j = 0; j < this.GetRefCount(); ++j)
            {
                if (i == j)
                    continue;
                ArrayList<Double> comDisList = new ArrayList<>();
                for (String com : Config.SigComList)
                {
                    ArrayList<Double> signal1 = refSigs.get(i).GetCom(com);
                    ArrayList<Double> signal2 = refSigs.get(j).GetCom(com);
                    // comDisList.add(NaiveDTW(signal1, signal2, Config.Penalization.get(com), Config.Threshold.get(com)));
                    comDisList.add(NaiveDTW(refSigs.get(i).GetCom(com), refSigs.get(j).GetCom(com), Config.Penalization.get(com), Config.Threshold.get(com)));
                }
                dis += Utils.GetMeanValue(comDisList);
            }
            refDis.add(dis);
        }
        this.templateIndex = Utils.GetMinIndex(refDis);
        this.templateSig = refSigs.get(this.templateIndex);
        // Utils.logger.log(Level.INFO, "Template index: {0}.", new Object[]{templateIndex});
    }

    /*
    @Description: Calculate the base value in signal component of signatures
     */
    public void CalcBaseDis()
    {
        // Utils.logger.log(Level.INFO, "calculating base distance");
        for (int k = 0; k < Config.SigComList.size(); ++k)
        {
            String com = Config.SigComList.get(k);
            ArrayList<Double> templateComList = new ArrayList<>();
            ArrayList<Double> maxComList = new ArrayList<>();
            ArrayList<Double> minComList = new ArrayList<>();
            ArrayList<Double> avgComList = new ArrayList<>();

            for (int i = 0; i < this.GetRefCount(); ++i)
            {
                if (i == this.templateIndex)
                    continue;
                ArrayList<Double> comi = this.refSigs.get(i).GetCom(com);
                Double templateComDis = NaiveDTW(comi, this.templateSig.GetCom(com),
                        Config.Penalization.get(com), Config.Threshold.get(com));
                templateComList.add(templateComDis);
                ArrayList<Double> comDisList = new ArrayList<>();
                for (int j = 0; j < this.GetRefCount(); ++j)
                {
                    if (i == j)
                        continue;
                    ArrayList<Double> comj = this.refSigs.get(j).GetCom(com);
                    comDisList.add(NaiveDTW(comi, comj, Config.Penalization.get(com), Config.Threshold.get(com)));
                }
                maxComList.add(Utils.GetMaxValue(comDisList));
                minComList.add(Utils.GetMinValue(comDisList));
                avgComList.add(Utils.GetMeanValue(comDisList));
            }
            if (Config.FeatureType.get(com).indexOf(Config.Template) != -1)
                this.baseValueMap.put(Config.Template+com, Utils.GetMeanValue(templateComList));
            if (Config.FeatureType.get(com).indexOf(Config.Max) != -1)
                this.baseValueMap.put(Config.Max+com, Utils.GetMeanValue(maxComList));
            if (Config.FeatureType.get(com).indexOf(Config.Min) != -1)
                this.baseValueMap.put(Config.Min+com, Utils.GetMeanValue(minComList));
            if (Config.FeatureType.get(com).indexOf(Config.Avg) != -1)
                this.baseValueMap.put(Config.Avg+com, Utils.GetMeanValue(avgComList));
            if (Config.FeatureType.get(com).indexOf(Config.Median) != -1)
                this.baseValueMap.put(Config.Median+com, Utils.GetMedianValue(avgComList));
            /*
            Utils.logger.log(Level.INFO, "Calculating signal: {0} {1} {2} {3} {4}",
                    new Object[]{com,
                    this.baseValueMap.get(Config.Template+com),
                    this.baseValueMap.get(Config.Max+com),
                    this.baseValueMap.get(Config.Min+com),
                    this.baseValueMap.get(Config.Avg+com)});
            */
        }
    }

    /*
    @Description: For given signature, calculate vector[] with normalization.
     */
    public ArrayList<Double> CalcDis(Signature signature)
    {
        ArrayList<Double> featureList = new ArrayList<>();
        for (String com : Config.SigComList)
        {
            ArrayList<Double> comSig = signature.GetCom(com);
            ArrayList<Double> comTemplate = this.templateSig.GetCom(com);
            Double templateComDis = NaiveDTW(comSig, comTemplate, Config.Penalization.get(com), Config.Threshold.get(com));
            ArrayList<Double> comDisList = new ArrayList<>();
            for (int i = 0; i < this.GetRefCount(); ++i)
            {
                ArrayList<Double> comI = this.refSigs.get(i).GetCom(com);
                Double dis = NaiveDTW(comSig, comI, Config.Penalization.get(com), Config.Threshold.get(com));
                comDisList.add(dis);
            }
            Double maxComDis = Utils.GetMaxValue(comDisList);
            Double minComDis = Utils.GetMinValue(comDisList);
            Double avgComDis = Utils.GetMeanValue(comDisList);
            Double medComDis = Utils.GetMedianValue(comDisList);
            if (Config.FeatureType.get(com).indexOf(Config.Template) != -1)
                featureList.add(templateComDis / this.baseValueMap.get(Config.Template+com));
            if (Config.FeatureType.get(com).indexOf(Config.Max) != -1)
                featureList.add(maxComDis / this.baseValueMap.get(Config.Max+com));
            if (Config.FeatureType.get(com).indexOf(Config.Min) != -1)
                featureList.add(minComDis / this.baseValueMap.get(Config.Min+com));
            if (Config.FeatureType.get(com).indexOf(Config.Avg) != -1)
                featureList.add(avgComDis / this.baseValueMap.get(Config.Avg+com));
            if (Config.FeatureType.get(com).indexOf(Config.Median) != -1)
                featureList.add(medComDis / this.baseValueMap.get(Config.Median+com));
        }
        return featureList;
    }

    public int GetRefCount()
    {
        return this.refSigs.size();
    }

    public Double NaiveDTW(ArrayList<Double> A, ArrayList<Double> B, Double p, Double t)
    {
        int size1 = A.size();
        int size2 = B.size();
        Double distance[][] = new Double[size1][size2];
        distance[0][0] = Math.abs(A.get(0) - B.get(0));
        for (int i = 1; i < size1; ++i)
            distance[i][0] = distance[i-1][0] + Math.abs(A.get(i) - B.get(0));
        for (int i = 1; i < size2; ++i)
            distance[0][i] = distance[0][i-1] + Math.abs(A.get(0) - B.get(i));
        for (int i = 1; i < size1; ++i)
        {
            for (int j = 1; j < size2; ++j)
            {
                if (1 == Config.DTWMethod)
                    distance[i][j] = Math.min(Math.min(distance[i-1][j], distance[i][j-1]), distance[i-1][j-1]) + Math.abs(A.get(i) - B.get(j));
                else if (2 == Config.DTWMethod)
                {
                    Double d1 = distance[i-1][j] + p;
                    Double d2 = distance[i][j-1] + p;
                    Double other = Math.abs(A.get(i) - B.get(j)) < t ? 0 : (Math.abs(A.get(i) - B.get(j)) - t);
                    Double d3 = distance[i-1][j-1] + other;
                    distance[i][j] = Math.min(d1, Math.min(d2, d3));
                }
                else if (3 == Config.DTWMethod)
                {
                    Double d1 = distance[i-1][j] + Math.abs(A.get(i) - B.get(j));
                    Double d2 = distance[i][j-1] + Math.abs(A.get(i) - B.get(j));
                    Double other = Math.abs(A.get(i) - B.get(j)) < t ? 0 : (Math.abs(A.get(i) - B.get(j) - t));
                    Double d3 = distance[i-1][j-1] + other;
                    distance[i][j] = Math.min(d1, Math.min(d2, d3));
                }
            }
        }

        return distance[size1-1][size2-1];
    }

    public ArrayList<Signature> refSigs;
    public Signature templateSig;
    public int templateIndex;
    public HashMap<String, Double> baseValueMap;
}
