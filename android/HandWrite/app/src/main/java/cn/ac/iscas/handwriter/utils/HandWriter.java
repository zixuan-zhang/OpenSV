package cn.ac.iscas.handwriter.utils;

import android.os.Environment;
import android.util.Log;
import android.util.SparseArray;
import android.database.Cursor;

import cn.ac.iscas.handwriter.Database;
import cn.ac.iscas.handwriter.MainActivity;
import cn.ac.iscas.handwriter.UnlockActivity;
import cn.ac.iscas.handwriter.views.SignaturePad;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class HandWriter
{
    public boolean register()
    {
        ArrayList<SparseArray<SignaturePad.MotionEventRecorder>> records=MainActivity.getrecords();
        if (records.size() < 5)
        {
            Log.i(Tag, "Registered signature size can not be less than 5");
            return false;
        }

        ArrayList<Signature> refSigs = buildSignatures(records);
        if (!signaturePersistence(records))
        {
            Log.e(Tag, "Signature persistence failed");
            return false;
        }

        // TODO: More reference signatures check
        _person = new Person(refSigs);
        System.out.println("Register success");
        return true;
    }

    public boolean check() {
        if (_person == null) {
            Log.e(Tag, "_person is null");
            return false;
        }
        if (UnlockActivity.classifier == null) {
            Log.e(Tag, "Classifier in UnlockActivity is null");
            return false;
        }

        ArrayList<SparseArray<SignaturePad.MotionEventRecorder>> records= UnlockActivity.getrecords();
        System.out.println("record size " + records.size());
        if (records.size() != 1)
        {
            Log.e(Tag, "Record size not equal to 1");
            return false;
        }

        ArrayList<Signature> signatures = buildSignatures(records);
        Signature signature = signatures.get(0);

        // Calculate features
        ArrayList<Double> feature = _person.CalcDis(signature);
        double[] array = new double[feature.size()];
        for (int i = 0; i < feature.size(); ++i) {
            array[i] = feature.get(i);
            Log.d(Tag, "Feature " + i + " is " + feature.get(i));
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
            Double classify = UnlockActivity.classifier.classifyInstance(instance);
            System.out.println("Score is " + classify);
            if (classify < 0.5) {
                return false;
            }
            else {
                return true;
            }
        }
        catch (Exception e) {
            Log.w(Tag, "Exception when classify instance " + e.toString());
            return false;
        }
    }

    public static HandWriter GetInstance() {
        return HandWrittenVerifierSingletonHolder.Instance;
    }

    private boolean signaturePersistence(ArrayList<SparseArray<SignaturePad.MotionEventRecorder>> records) {
        if (_database == null) {
            Log.e(Tag, "Database is null");
            return false;
        }

        // First delete old signature records if exist, then insert new data.
        _database.deleteFromSignatureRecord();
        for (int sIndex = 0; sIndex < records.size(); ++sIndex) {
            SparseArray<SignaturePad.MotionEventRecorder> record = records.get(sIndex);
            for (int i = 0; i < record.size(); ++i) {
                SignaturePad.MotionEventRecorder r = record.get(i, null);
                if (r != null) {
                    _database.insertData(sIndex, (double)r.getTime(), (double)r.getX(),
                            (double)r.getY(), (double)r.getZ());
                }
            }
        }
        return true;
    }

    private ArrayList<Signature> buildSignatures(ArrayList<SparseArray<SignaturePad.MotionEventRecorder>> records) {
        ArrayList<Signature> refSigs = new ArrayList<>();

        for (int sIndex = 0; sIndex < records.size(); ++sIndex) {
            SparseArray<SignaturePad.MotionEventRecorder> record = records.get(sIndex);
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

            Signature signature = buildSignature(T, X, Y, P);
            refSigs.add(signature);
        }
        return refSigs;
    }

    private Signature buildSignature(ArrayList<Double> T, ArrayList<Double> X,
                                     ArrayList<Double> Y, ArrayList<Double> P) {
        Signature signature = new Signature();

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

        return signature;
    }

    private static class HandWrittenVerifierSingletonHolder {
        public static final HandWriter Instance = new HandWriter();
    }

    private HandWriter() {
        try
        {
            sdLog = new FileOutputStream(new File(Environment.getExternalStorageDirectory().getCanonicalPath(), "writer.log"), true);
        }
        catch (IOException e)
        {
            Log.e(Tag, "Error when create sdLog");
        }

        try
        {
            _person = null;

            // load signatures from database if exist.
            if (MainActivity.database != null) {
                _database = MainActivity.database;
                Log.d(Tag, "MainActivity database is not null");
                sdLog.write("MainActivity database is not null".getBytes());
            }
            if (UnlockActivity.database != null) {
                Log.d(Tag, "UnlockActivity database is not null");
                _database = UnlockActivity.database;
                sdLog.write("UnlockActivity database is not null".getBytes());
            }
            if (_database == null) {
                Log.e(Tag, "Database is null when initialize HandWriter");
                sdLog.write("Database is null when initialize HandWriter".getBytes());
            }
            if (_database != null && _database.searchTotalCount() > 0) {
                Log.i(Tag, "Found signature records when initialize HandWriter, reload them");
                sdLog.write("Found signature records when initialize HandWriter, reload them".getBytes());
                ArrayList<Signature> signatures = new ArrayList<>();
                for (int sigId = 0; sigId < 5; ++sigId) {
                    Cursor cursor = _database.searchBySignatureId(sigId);
                    ArrayList<Double> T = new ArrayList<>();
                    ArrayList<Double> X = new ArrayList<>();
                    ArrayList<Double> Y = new ArrayList<>();
                    ArrayList<Double> P = new ArrayList<>();

                    if (cursor.moveToFirst()) {
                        while (cursor.isAfterLast() == false) {
                            T.add(cursor.getDouble(cursor.getColumnIndex("timestamp")));
                            X.add(cursor.getDouble(cursor.getColumnIndex("x")));
                            Y.add(cursor.getDouble(cursor.getColumnIndex("y")));
                            P.add(cursor.getDouble(cursor.getColumnIndex("y")));
                            cursor.moveToNext();
                        }
                        signatures.add(buildSignature(T, X, Y, P));
                    }
                }
                _person = new Person(signatures);
            }
            else {
                Log.i(Tag, "No signature records in database");
                sdLog.write("No signature records in database".getBytes());
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }

    }

    private Person _person = null;
    private final String Tag = "HandWriter";
    private Database _database = null;
    public static FileOutputStream sdLog;
}
