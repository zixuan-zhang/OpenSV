/**
 * Created by zixuan on 2016/9/3.
 */

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Level;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
//import net.sf.javaml.core.SparseInstance;
//import net.sf.javaml.tools.ListTools;
//import weka.classifiers.trees.RandomForest;
//import weka.core.DenseInstance;
//import weka.core.Instances;


public class Driver {


    public Driver() throws Exception
    {
        /*
        if (!Init())
        {
            throw new Exception("Error when init");
        }
        */

        if (Config.Database.equals("SVC"))
            GetSVCSignatures();
        else if (Config.Database.equals(Config.SUSIGDatabase) || Config.Database.equals(Config.SelfDatabase))
            GetSignatures(Config.Database);
        else
            throw new Exception("DataSet not supported");

        // Utils.logger.log(Level.INFO, "Train set count: {0}", this.trainSets.size());
        // Utils.logger.log(Level.INFO, "Test set count: {0}", this.testSets.size());

        ArrayList<ArrayList<Double>> genuineX = new ArrayList<>();
        ArrayList<ArrayList<Double>> forgeryX = new ArrayList<>();
        for (int i = 0; i < this.trainSets.size(); ++i)
        {
            // Utils.logger.log(Level.INFO, "Training Process: {0}", i);
            PersonTrain personTrain = new PersonTrain(this.trainSets.get(i).GetGenuine(),
                    this.trainSets.get(i).GetForgery());
            ArrayList<ArrayList<Double>> genuine = new ArrayList<>();
            ArrayList<ArrayList<Double>> forgery = new ArrayList<>();
            personTrain.CalculateTrainSet(genuine, forgery);
            genuineX.addAll(genuine);
            forgeryX.addAll(forgery);
        }

        // Utils.logger.log(Level.INFO, "GenuineX count: {0}", genuineX.size());
        // Utils.logger.log(Level.INFO, "ForgeryX count: {0}", forgeryX.size());

        classifier = new KNearestNeighbors(5);
        Dataset dataset = new DefaultDataset();
        for (int i = 0; i < genuineX.size(); ++i)
        {
            double[] array = genuineX.get(i).stream().mapToDouble(d->d).toArray();
            DenseInstance instance = new DenseInstance(array, 1);
            dataset.add(instance);
        }
        for (int i = 0; i < forgeryX.size(); ++i)
        {
            double[] array = forgeryX.get(i).stream().mapToDouble(d->d).toArray();
            DenseInstance instance = new DenseInstance(array, 0);
            dataset.add(instance);
        }

        // Utils.logger.log(Level.INFO, "trainDataSet size: {0}", dataset.size());
        // Fit classifier
        classifier.buildClassifier(dataset);
    }

    public void test()
    {
        // Utils.logger.log(Level.INFO, "Start test");

        int genuineCount = 0;
        int forgeryCount = 0;
        int falseRejectCount = 0;
        int falseAcceptCount = 0;

        for (PersonSignatures personSignature : this.testSets)
        {
            PersonTest personTest = new PersonTest(new ArrayList<Signature>(personSignature.GetGenuine().subList(0, Config.RefCount)));
            ArrayList<Signature> genuineSet = personSignature.GetGenuine();
            ArrayList<Signature> forgerySet = personSignature.GetForgery();
            genuineCount += genuineSet.size();
            forgeryCount += forgerySet.size();
            for (int i = 0; i < genuineSet.size(); ++i)
            {
                ArrayList<Double> feature = personTest.CalcDis(genuineSet.get(i));
                double[] array = feature.stream().mapToDouble(d->d).toArray();
                DenseInstance instance = new DenseInstance(array, 1);
                Object predictedValue = this.classifier.classify(instance);
                if (!predictedValue.equals(instance.classValue())) {
                    // Utils.logger.log(Level.INFO, "FalseReject: {0} {1}", new Object[]{personSignature.GetUserName(), i});
                    falseRejectCount += 1;
                }
            }
            for (int i = 0; i < forgerySet.size(); ++i)
            {
                ArrayList<Double> feature = personTest.CalcDis(forgerySet.get(i));
                double[] array = feature.stream().mapToDouble(d->d).toArray();
                DenseInstance instance = new DenseInstance(array, 0);
                Object predictedValue = this.classifier.classify(instance);
                if (!predictedValue.equals(instance.classValue())) {
                    // Utils.logger.log(Level.INFO, "FalseAccepted: {0} {1}", new Object[]{personSignature.GetUserName(), i});
                    falseAcceptCount += 1;
                }
            }
        }

        Config.ShowConfig();
        double falseAcceptRate = 1 - falseRejectCount/(float)genuineCount;
        double falseRejectRate = 1 - falseAcceptCount/(float)forgeryCount;
        double avg = (falseAcceptRate + falseRejectRate) / 2;
        Utils.logger.log(Level.INFO, "Result : FalseAcceptRate {0} : FalseRejectRate {1} : Avg {2}", new Object[]{falseAcceptRate, falseRejectRate, avg});
    }

    private void GetSVCSignatures()
    {
        // Utils.logger.log(Level.INFO, "Getting signatures from svc database");
        ArrayList<PersonSignatures> totalSets = new ArrayList<>();
        for (int i = 1; i <= 40; ++i)
        {
            ArrayList<Signature> signatures = new ArrayList<>();
            for (int j = 1; j <= 40; ++j)
            {
                String fileName = "U"+i+"S"+j+".TXT";
                String filePath = Paths.get(Config.SvcDataPath, fileName).toString();
                int label = j <= 20 ? 1 : 0;
                try
                {
                    Signature signature = GetSignatureFromSvcFile(filePath, label);
                    signatures.add(signature);
                }
                catch (IOException e)
                {
                    // Utils.logger.log(Level.INFO, "Exception: {0}", e);
                }
            }
            PersonSignatures personSignatures = new PersonSignatures(String.valueOf(i),
                    new ArrayList<Signature>(signatures.subList(0, 20)),
                    new ArrayList<Signature>(signatures.subList(20, 40)));
            totalSets.add(personSignatures);
        }

        // Initialize trainSets and testSets
        this.trainSets = new ArrayList<>(totalSets.subList(0, (int) (totalSets.size() * Config.TrainSetRate)));
        this.testSets =  new ArrayList<>(totalSets.subList((int)(totalSets.size()*Config.TrainSetRate), totalSets.size()));
    }

    private Signature GetSignatureFromSvcFile(String filePath, int label) throws IOException
    {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);
        lines.remove(0);
        String userName = path.getFileName().toString().split("S")[0].split("U")[1];
        Signature signature = new Signature(userName, label);
        ArrayList<Double> X = new ArrayList<>();
        ArrayList<Double> Y = new ArrayList<>();
        ArrayList<Double> T = new ArrayList<>();
        ArrayList<Double> P = new ArrayList<>();
        for (String line : lines)
        {
            String items[] = line.split(" ");
            X.add(Double.valueOf(items[0]));
            Y.add(Double.valueOf(items[1]));
            T.add(Double.valueOf(items[2])/10.0);
            P.add(Double.valueOf(items[6]));
        }

        // Pre-process
        Processor.size_normalization(X, Y, Config.PreWidth, Config.PreHeight);
        Processor.location_normalization(X, Y);

        ArrayList<Double> VX = Processor.calculate_delta(X, T);
        ArrayList<Double> VY = Processor.calculate_delta(Y, T);

        signature.SetCom("X", X);
        signature.SetCom("Y", Y);
        signature.SetCom("T", T);
        signature.SetCom("P", P);
        signature.SetCom("VX", VX);
        signature.SetCom("VY", VY);
        return signature;
    }

    /*
    @Desc: Get signatures from data. The data will write to this.trainSets and this.testSets.
     */
    private void GetSignatures(String database) {
        // Utils.logger.log(Level.INFO, "Getting signatures from {0}", database);
        ArrayList<PersonSignatures> totalSets = new ArrayList<>();

        String forgeryFolderPath = new String();
        String genuineFolderPath = new String();
        if (database.equals(Config.SUSIGDatabase)) {
            forgeryFolderPath = Paths.get(Config.SusigDataPath, Config.VisualSubCorpus, Config.Forgery).toString();
            genuineFolderPath = Paths.get(Config.SusigDataPath, Config.VisualSubCorpus, Config.Genuine).toString();
        }
        else if (database.equals(Config.SelfDatabase))
        {
            forgeryFolderPath = Paths.get(Config.SelfDataPath, Config.Forgery).toString();
            genuineFolderPath = Paths.get(Config.SelfDataPath, Config.Genuine).toString();
        }
        HashMap<String, ArrayList<Signature>> userToForgery = GetUserToSigsMap(database, forgeryFolderPath, 0);
        HashMap<String, ArrayList<Signature>> userToGenuine = GetUserToSigsMap(database, genuineFolderPath, 1);
        Set<String> users = new HashSet<>(userToForgery.keySet());
        users.addAll(userToGenuine.keySet());

        for (String user : users)
        {
            PersonSignatures personSignature = new PersonSignatures(user, userToGenuine.get(user), userToForgery.get(user));
            totalSets.add(personSignature);
        }

        // Initialize trainSets and testSets
        this.trainSets = new ArrayList<>(totalSets.subList(0, (int) (totalSets.size() * Config.TrainSetRate)));
        this.testSets =  new ArrayList<>(totalSets.subList((int)(totalSets.size()*Config.TrainSetRate), totalSets.size()));
    }

    private HashMap<String, ArrayList<Signature>> GetUserToSigsMap(String database, String folderPath, int label)
    {
        // Utils.logger.log(Level.INFO, "Getting Signatures from {0}, label is {1}", new Object[]{folderPath, label});
        HashMap<String, ArrayList<Signature>> userToSignatures = new HashMap<>();
        File folder = new File(folderPath);
        File[] fileList = folder.listFiles();
        for (File file : fileList)
        {
            String userName = ExtractUserNameFromFile(database, file);
            if (!userToSignatures.containsKey(userName))
                userToSignatures.put(userName, new ArrayList<>());
            Signature signature = GetSignatureFromFile(database, file.getPath(), label);
            userToSignatures.get(userName).add(signature);
        }
        return userToSignatures;
    }

    private String ExtractUserNameFromFile(String database, File file)
    {
        String userName = userName = file.getName().split("_")[0];
        return userName;
    }

    private Signature GetSignatureFromFile(String database, String filePath, int label)
    {
        try {
            if (database.equals(Config.SUSIGDatabase)) {
                return GetSignatureFromSUSIGFile(filePath, label);
            }
            else if (database.equals(Config.SelfDatabase)) {
                return GetSignatureFromSelfFile(filePath, label);
            }
        } catch (Exception e) {
            // Utils.logger.log(Level.INFO, "Exception when get signature from {0} file. {1}", new Object[]{database, e.toString()});
        }
        return null;
    }

    private Signature GetSignatureFromSelfFile(String filePath, int label) throws IOException
    {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);
        String userName = path.getFileName().toString().split("_")[0];
        Signature signature = new Signature(userName, label);
        ArrayList<Double> T = new ArrayList<>();
        ArrayList<Double> X = new ArrayList<>();
        ArrayList<Double> Y = new ArrayList<>();
        ArrayList<Double> P = new ArrayList<>();
        for (String line : lines)
        {
            if (line.isEmpty())
                continue;
            String items[] = line.split(" ");
            T.add(Double.valueOf(items[0]));
            X.add(Double.valueOf(items[1]));
            Y.add(Double.valueOf(items[2]));
            P.add(Double.valueOf(items[3]));
        }

        Double maxY = Utils.GetMaxValue(Y);
        for (int i = 0; i < Y.size(); ++i)
            Y.set(i, maxY - Y.get(i));

        // Pre-process
        Processor.size_normalization(X, Y, Config.PreWidth, Config.PreHeight);
        Processor.location_normalization(X, Y);

        ArrayList<Double> VX = Processor.calculate_delta(X, T);
        ArrayList<Double> VY = Processor.calculate_delta(Y, T);

        signature.SetCom("X", X);
        signature.SetCom("Y", Y);
        signature.SetCom("T", T);
        signature.SetCom("P", P);
        signature.SetCom("VX", VX);
        signature.SetCom("VY", VY);
        return signature;
    }

    private Signature GetSignatureFromSUSIGFile(String filePath, int label) throws IOException
    {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);
        lines.remove(0);
        lines.remove(0);
        String userName = path.getFileName().toString().split("_")[0];
        Signature signature = new Signature(userName, label);
        ArrayList<Double> X = new ArrayList<>();
        ArrayList<Double> Y = new ArrayList<>();
        ArrayList<Double> T = new ArrayList<>();
        ArrayList<Double> P = new ArrayList<>();
        for (String line : lines)
        {
            String items[] = line.split(" ");
            X.add(Double.valueOf(items[0]));
            Y.add(Double.valueOf(items[1]));
            T.add(Double.valueOf(items[2]));
            P.add(Double.valueOf(items[3]));
        }

        // Pre-process
        Processor.size_normalization(X, Y, Config.PreWidth, Config.PreHeight);
        Processor.location_normalization(X, Y);

        ArrayList<Double> VX = Processor.calculate_delta(X, T);
        ArrayList<Double> VY = Processor.calculate_delta(Y, T);

        signature.SetCom("X", X);
        signature.SetCom("Y", Y);
        signature.SetCom("T", T);
        signature.SetCom("P", P);
        signature.SetCom("VX", VX);
        signature.SetCom("VY", VY);
        return signature;
    }

    private ArrayList<PersonSignatures> trainSets;

    private ArrayList<PersonSignatures> testSets;

    private Classifier classifier;

    private FileHandler fh;
}
