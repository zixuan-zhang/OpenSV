import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by zixuan on 2016/9/1.
 */
public class Program {
    public static void main(String[] args)
    {
        FinalTest();
        // TrainTest();
    }

    private static void FinalTest()
    {
        Utils.Init();
        try
        {
            Driver driver = new Driver();
            driver.test();
        }
        catch (Exception e)
        {
            Utils.logger.log(Level.INFO, "Exception when running : {0}", e.toString());
            e.printStackTrace();
        }
    }

    private static void TrainTest()
    {
        Utils.Init();

        final Double ThresholdMin = 2.0;
        final Double ThresholdMax = 8.0;

        ArrayList<ArrayList<String>> sigComLists = new ArrayList<>();
        /*
        sigComLists.add(new ArrayList<>(Arrays.asList("X")));
        sigComLists.add(new ArrayList<>(Arrays.asList("Y")));
        sigComLists.add(new ArrayList<>(Arrays.asList("VX")));
        */
        sigComLists.add(new ArrayList<>(Arrays.asList("VY")));
        /*
        sigComLists.add(new ArrayList<>(Arrays.asList("X", "Y")));
        sigComLists.add(new ArrayList<>(Arrays.asList("VX", "VY")));
        sigComLists.add(new ArrayList<>(Arrays.asList("X", "VX")));
        sigComLists.add(new ArrayList<>(Arrays.asList("Y", "VY")));
        sigComLists.add(new ArrayList<>(Arrays.asList("X", "Y", "VY")));
        sigComLists.add(new ArrayList<>(Arrays.asList("X", "VX", "VY")));
        sigComLists.add(new ArrayList<>(Arrays.asList("X", "Y", "VX", "VY")));
        */

        try
        {
            for (int i = 0; i < sigComLists.size(); ++i)
            {
                ArrayList<String> sigComList = sigComLists.get(i);
                Config.SigComList = sigComList;
                for (Double xp = ThresholdMin; xp <= ThresholdMax; xp += 1)
                {
                    Config.Penalization.put("X", xp);
                    for (Double xt = ThresholdMin; xt <= ThresholdMax; xt += 1)
                    {
                        Config.Threshold.put("X", xt);
                        for (Double yp = ThresholdMin; yp <= ThresholdMax; yp += 1)
                        {
                            Config.Penalization.put("Y", yp);
                            for (Double yt = ThresholdMin; yt <= ThresholdMax; yt += 1)
                            {
                                Config.Threshold.put("Y", yt);
                                for (Double vxp = ThresholdMin; vxp <= ThresholdMax; vxp += 1)
                                {
                                    Config.Penalization.put("VX", vxp);
                                    for (Double vxt = ThresholdMin; vxt <= ThresholdMax; vxt += 1)
                                    {
                                        Config.Threshold.put("VX", vxt);
                                        for (Double vyp = ThresholdMin; vyp <= ThresholdMax; vyp += 1)
                                        {
                                            Config.Penalization.put("VY", vyp);
                                            for (Double vyt = ThresholdMin; vyt <= ThresholdMax; vyt += 1)
                                            {
                                                Config.Threshold.put("VY", vyt);
                                                Driver driver = new Driver();
                                                driver.test();
                                                if (sigComList.indexOf("VY") == -1)
                                                    break;
                                            }
                                            if (sigComList.indexOf("VY") == -1)
                                                break;
                                        }
                                        if (sigComList.indexOf("VX") == -1)
                                            break;
                                    }
                                    if (sigComList.indexOf("VX") == -1)
                                        break;
                                }
                                if (sigComList.indexOf("Y") == -1)
                                    break;
                            }
                            if (sigComList.indexOf("Y") == -1)
                                break;
                        }
                        if (sigComList.indexOf("X") == -1)
                            break;
                    }
                    if (sigComList.indexOf("X") == -1)
                        break;
                }
            }
        }
        catch (Exception e)
        {
            Utils.logger.log(Level.INFO, "{0}", e);
        }
    }

}
