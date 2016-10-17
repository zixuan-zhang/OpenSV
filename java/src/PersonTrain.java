/**
 * Created by zixuan on 2016/9/2.
 */

import java.util.ArrayList;
import java.util.logging.Level;

public class PersonTrain extends  Person{
    public PersonTrain(ArrayList<Signature> genuineSigs, ArrayList<Signature> forgerySigs)
    {
        super(new ArrayList<>(genuineSigs.subList(0, Config.RefCount)));
        this.genuineSigs = new ArrayList<>(genuineSigs.subList(Config.RefCount, genuineSigs.size()));
        this.forgerySigs = forgerySigs;

        // Utils.logger.log(Level.INFO, "Reference signature count: {0}", this.GetRefCount());
        // Utils.logger.log(Level.INFO, "Genuine signature count: {0}", this.genuineSigs.size());
        // Utils.logger.log(Level.INFO, "Forgery signature count: {0}", this.forgerySigs.size());
    }

    public void CalculateTrainSet(ArrayList<ArrayList<Double>> genuineFeaturesList,
                                  ArrayList<ArrayList<Double>> forgeryFeaturesList)
    {
        genuineFeaturesList.clear();
        forgeryFeaturesList.clear();
        for (int i = 0; i < this.genuineSigs.size(); ++i)
        {
            ArrayList<Double> genuineFeature = this.CalcDis(this.genuineSigs.get(i));
            // Utils.logger.log(Level.INFO, "Genuine vector: {0}", genuineFeature);
            genuineFeaturesList.add(genuineFeature);
        }

        for (int i = 0; i < this.forgerySigs.size(); ++i)
        {
            ArrayList<Double> forgeryFeature = this.CalcDis(this.forgerySigs.get(i));
            // Utils.logger.log(Level.INFO, "Forgery vector: {0}", forgeryFeature);
            forgeryFeaturesList.add(forgeryFeature);
        }
    }

    private ArrayList<Signature> genuineSigs;

    private ArrayList<Signature> forgerySigs;
}
