package cn.ac.iscas.handwriter.utils;

import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.util.SparseArray;

import cn.ac.iscas.handwriter.MainActivity;
import cn.ac.iscas.handwriter.UnlockActivity;
import cn.ac.iscas.handwriter.views.SignaturePad;

import java.util.ArrayList;
import java.util.logging.Level;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class HandWriter
{
    public boolean register()
    {
        ArrayList<SparseArray<SignaturePad.MotionEventRecorder>> records=MainActivity.getrecords();
        System.out.println(records.size());

        if (records.size() < 5)
        {
            Utils.logger.log(Level.WARNING, "Registed signature size can not be less than 5");
            System.out.println("Registed signature size can not be less than 5");
            return false;
        }

        ArrayList<Signature> refSigs = buildSignatures(records);
        // TODO: More reference signatures check
        // Initialize _person
        _person = new Person(refSigs);
        _isEnable = true;
        System.out.println("Register success");
        return true;
    }

    public boolean check(){

        if (!_isEnable || _person == null)
        {
            System.out.println("_isEnable is false or _person is null");
            return false;
        }

        ArrayList<SparseArray<SignaturePad.MotionEventRecorder>> records= UnlockActivity.getrecords();
        System.out.println("record size " + records.size());
        if (records.size() != 1)
        {
            return false;
        }

        ArrayList<Signature> signatures = buildSignatures(records);
        Signature signature = signatures.get(0);

        // Calculate features
        ArrayList<Double> feature = _person.CalcDis(signature);
        double[] array = new double[feature.size()];
        for (int i = 0; i < feature.size(); ++i) {
            array[i] = feature.get(i);
            System.out.println(feature.get(i));
        }

        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < Config.SigComList.size() * Config.GetFeatureType().size() + 1 ; ++i)
        {
            attributes.add(new Attribute(String.valueOf(i)));
        }
        Instances instances = new Instances("Instances", attributes, 0);
        instances.setClassIndex(instances.numAttributes() - 1);

        DenseInstance instance = new DenseInstance(1.0, array);
        instance.setDataset(instances);
        try {
            Double classify = MainActivity.classifier.classifyInstance(instance);
            System.out.println("Score is " + classify);
            if (classify < 0.5) {
                return false;
            }
            else {
                return true;
            }
        }
        catch (Exception e) {
            Utils.logger.log(Level.WARNING, "Exception when classify instance {0}", e);
            return false;
        }
    }



    public static HandWriter GetInstance() {
        return HandWrittenVerifierSingletonHolder.Instance;
    }

    private ArrayList<Signature> buildSignatures(ArrayList<SparseArray<SignaturePad.MotionEventRecorder>> records) {
        ArrayList<Signature> refSigs = new ArrayList<>();

        for (int sIndex = 0; sIndex < records.size(); ++sIndex) {
            SparseArray<SignaturePad.MotionEventRecorder> record = records.get(sIndex);
            Signature signature = new Signature();
            ArrayList<Double> T = new ArrayList<>();
            ArrayList<Double> X = new ArrayList<>();
            ArrayList<Double> Y = new ArrayList<>();
            ArrayList<Double> P = new ArrayList<>();
            for (int i = 0; i < record.size(); ++i) {
                SignaturePad.MotionEventRecorder r = record.get(i, null);
                if (r != null) {
                    T.add((double) r.getTime());
                    X.add((double) r.getX());
                    Y.add((double) r.getY());
                    P.add((double) r.getZ());
                }
            }

            // Need to re-calculate Y dimension.
            Double maxY = Utils.GetMaxValue(Y);
            for (int i = 0; i < Y.size(); ++i)
                Y.set(i, maxY - Y.get(i));

            // Pre-process
            Processor.size_normalization(X, Y, Config.PreWidth, Config.PreHeight);
            Processor.location_normalization(X, Y);

            ArrayList<Double> VX = Processor.calculate_delta(X, T);
            ArrayList<Double> VY = Processor.calculate_delta(Y, T);

            // Construct signature object
            signature.SetCom("X", X);
            signature.SetCom("Y", Y);
            signature.SetCom("T", T);
            signature.SetCom("P", P);
            signature.SetCom("VX", VX);
            signature.SetCom("VY", VY);

            refSigs.add(signature);
        }
        return refSigs;
    }

    private static class HandWrittenVerifierSingletonHolder {
        public static final HandWriter Instance = new HandWriter();
    }

    private HandWriter() {
        _isEnable = false;
        _person = null;
        if (MainActivity.classifier != null)
        {
            _isEnable = true;
        }
    }

    private boolean _isEnable;
    private Person _person;
}
