package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import weka.core.Instances;
import weka.filters.unsupervised.instance.Randomize;

public class MainHW3 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    /**
     * Finds the best hyper parameters (K – number of neighbors, Lp distance measure, weighting scheme)
     * using 10-folds cross validation, for a given data set.
     *
     * @param maxK  - try k's in range [1..maxK]
     * @param set_p - a given set of values
     * @param data
     * @return [k, p, prediction type (0 - uniform, 1 - weighted), avg error]
     */
    private static double[] searchHyperParameters(int maxK, double[] set_p, Instances data) throws Exception {
        Knn knn = new Knn();
        knn.buildClassifier(data);
        double currentError;
        double[] bestHyperParameters = new double[4];
        bestHyperParameters[3] = Double.MAX_VALUE;


        for (int k = 1; k <= maxK; k++) {
            knn.setK(k);
            for (double p : set_p) {
                knn.setP(p);
                for (Knn.PredictionType predictionType : Knn.PredictionType.values()) {
                    knn.setPredictionType(predictionType);
                    currentError = knn.crossValidationError(data, 10);
                    if (currentError < bestHyperParameters[3]) {
                        bestHyperParameters[0] = k;
                        bestHyperParameters[1] = p;
                        bestHyperParameters[2] = (predictionType == Knn.PredictionType.Uniform) ? 0 : 1;
                        bestHyperParameters[3] = currentError;
                    }

                }
            }
        }

        return bestHyperParameters;
    }

    private static void appendSectorsCrossValidation(StringBuilder results,Instances randomizedScaledData,
                                                     int[] set_folds, double[] bestParamScaled) throws Exception {
        Knn knn = new Knn();
        knn.buildClassifier(randomizedScaledData);
        knn.setK((int) bestParamScaled[0]);
        knn.setP(bestParamScaled[1]);
        knn.setPredictionType((bestParamScaled[2] == 0) ?
                Knn.PredictionType.Uniform : Knn.PredictionType.Weighted);
        knn.setMeasureTime(true);
        long predictionRunTime;
        double error;
        String dc;

        for (int folds : set_folds) {
            results.append("----------------------------\n" +
                    "Results for " + folds + " folds: \n" +
                    "----------------------------\n");
            for (Knn.DistanceCheck distanceCheck : Knn.DistanceCheck.values()) {
                dc = (distanceCheck == Knn.DistanceCheck.Regular) ? "regular" : "efficient";
                knn.setDistanceCheck(distanceCheck);
                error = knn.crossValidationError(randomizedScaledData, folds);
                predictionRunTime = knn.getPredictionRunTime();
                results.append("Cross validation error of " + dc + " knn on auto_price dataset is " + error +
                        "\nThe average elapsed time is: " + (predictionRunTime / folds) + " ns" +
                        "\nThe total elapsed time is: " + predictionRunTime + " ns\n\n");
            }
        }
    }

    private static void appendSectorBestParams(StringBuilder results,
                                               double[] bestParamUnscaled, double[] bestParamScaled) {
        results.append("----------------------------\n" +
                "Results for original dataset: " +
                "\n----------------------------\n" +
                "Cross validation error with K = " + (int) bestParamUnscaled[0] +
                ", lp = " + (int) bestParamUnscaled[1] +
                ", majority function = " + ((bestParamUnscaled[2] == 0) ? "uniform" : "weighted") +
                ",\nfor auto_price data is: " + bestParamUnscaled[3] + "\n\n" +
                "----------------------------\n" +
                "Results for scaled dataset: \n" +
                "----------------------------\n" +
                "Cross validation error with K = " + (int) bestParamScaled[0] +
                ", lp = " + (int) bestParamScaled[1] +
                ", majority function = " + ((bestParamScaled[2] == 0) ? "uniform" : "weighted") +
                ",\nfor auto_price data is: " + bestParamScaled[3] + "\n\n");
    }


    public static void main(String[] args) throws Exception {
        Instances data = loadData("auto_price.txt");
        Instances randomizedData = loadData("auto_price.txt");
        randomizedData.randomize(new java.util.Random(4));
        Instances randomizedScaledData = new FeatureScaler().scaleData(randomizedData);
        StringBuilder results = new StringBuilder("");

        int maxK = 20;
        double[] set_p = {1, 2, 3, Double.MAX_VALUE};
        int[] set_folds = {data.numInstances(), 50, 10, 5, 3};

        // returns [k, p, prediction type (0 - uniform, 1 - weighted), avg error]
        double[] bestParamUnscaled = searchHyperParameters(maxK, set_p, randomizedData);
        double[] bestParamScaled = searchHyperParameters(maxK, set_p, randomizedScaledData);

        appendSectorBestParams(results, bestParamUnscaled, bestParamScaled);
        appendSectorsCrossValidation(results, randomizedScaledData, set_folds, bestParamScaled);

        System.out.println(results.toString());

    }

}
