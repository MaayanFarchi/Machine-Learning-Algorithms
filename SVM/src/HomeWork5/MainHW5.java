package HomeWork5;

import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class MainHW5 {

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
     * computes the TPR and FPR for each kernel
     *
     * @param degrees
     * @param gammas
     * @return a 2-D array of kernels data: [0-poly | 1-RBF, degree | gamma, TPR, FPR]
     */
    public static double[][] calcKernelsMeasures(SplittedData data, double[] degrees, double[] gammas) throws Exception {
        double[][] results = new double[degrees.length + gammas.length][4];
        int[] cm;  // [TP, FP, TN, FN]
        SVM svm = new SVM();
        int i = 0;
        double[] rates;

        PolyKernel polyKernel = new PolyKernel();
        for (double d : degrees) {
            polyKernel.setExponent(d);
            svm.setKernel(polyKernel);
            svm.buildClassifier(data.train);
            cm = svm.calcConfusion(data.test);
            rates = calcRates(cm);
            results[i][0] = 0;
            results[i][1] = d;
            results[i][2] = rates[0]; // TPR
            results[i][3] = rates[1]; // FPR
            i++;
        }

        RBFKernel rbfKernel = new RBFKernel();
        for (double g : gammas) {
            rbfKernel.setGamma(g);
            svm.setKernel(rbfKernel);
            svm.buildClassifier(data.train);
            cm = svm.calcConfusion(data.test);
            rates = calcRates(cm);
            results[i][0] = 1;
            results[i][1] = g;
            results[i][2] = rates[0]; // TPR
            results[i][3] = rates[1]; // FPR
            i++;
        }
        return results;
    }

    public static int findBestKernelIndex(double[][] kernelsData, double alpha) {
        double currentScore;
        double bestScore = Double.MIN_VALUE;
        int bestIndex = 0;

        for (int i = 0; i < kernelsData.length; i++) {
            currentScore = alpha * kernelsData[i][2] - kernelsData[i][3];
            if (currentScore > bestScore) {
                bestScore = currentScore;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    /**
     * computes the TPR and FPR for each C
     *
     * @param data
     * @param kernelInfo
     * @param c_i
     * @param c_j
     * @return a 2-D array of C data: [C, TPR, FPR]
     */
    private static double[][] calcCMeasures(SplittedData data, double[] kernelInfo, double[] c_i, double[] c_j) throws Exception {
        double[][] results = new double[c_i.length * c_j.length][3];
        int[] cm;  // [TP, FP, TN, FN]
        int k = 0;
        SVM svm = new SVM();
        double c;
        double[] rates;

        if (kernelInfo[0] == 0) {
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(kernelInfo[1]);
            svm.setKernel(kernel);
        } else {
            RBFKernel kernel = new RBFKernel();
            kernel.setGamma(kernelInfo[1]);
            svm.setKernel(kernel);
        }

        for (double i : c_i) {
            for (double j : c_j) {
                c = Math.pow(10, i) * j / 3;
                svm.setC(c);
                svm.buildClassifier(data.train);
                cm = svm.calcConfusion(data.test);
                rates = calcRates(cm);
                results[k][0] = c;
                results[k][1] = rates[0]; // TPR
                results[k][2] = rates[1]; // FPR
                k++;
            }
        }
        return results;
    }

    private static double[] calcRates(int[] cm) {

        double[] rates = new double[2];
        // TPR = TP / (TP + FN)
        rates[0] = (cm[0] != 0) ? (double) cm[0] / (cm[0] + cm[3]) : 0;
        // FPR = FP / (FP + TN)
        rates[1] = (cm[1] != 0) ? (double) cm[1] / (cm[1] + cm[2]) : 0;

        return rates;
    }

    private static void appendSectionKernel(StringBuilder str, double[][] kernelsMeasures, int bestKernelIndex, double a) {
        String kType;

        for (int i = 0; i < kernelsMeasures.length; i++) {
            kType = (kernelsMeasures[i][0] == 0) ? "Poly Kernel with degree " : "RBF Kernel with gamma ";
            str.append(
                    "For " + kType + kernelsMeasures[i][1] + " the rates are:\n" +
                            "\tTPR = " + kernelsMeasures[i][2] + "\n" +
                            "\tFPR = " + kernelsMeasures[i][3] + "\n\n"
            );
        }

        kType = (kernelsMeasures[bestKernelIndex][0] == 0) ? "Poly Kernel with degree " : "RBF Kernel with gamma ";
        str.append(
                "The best kernel is: " + kType + kernelsMeasures[bestKernelIndex][1] + " and score of: " +
                        (a * kernelsMeasures[bestKernelIndex][2] - kernelsMeasures[bestKernelIndex][3]) + "\n\n"
        );
    }

    private static void appendSectionC(StringBuilder str, double[][] cMeasures) {
        for (int i = 0; i < cMeasures.length; i++) {
            str.append(
                    "For C: " + cMeasures[i][0] + ", the rates are:\n" +
                            "\tTPR = " + cMeasures[i][1] + "\n" +
                            "\tFPR = " + cMeasures[i][2] + "\n\n"
            );
        }
    }

    public static void main(String[] args) throws Exception {
        Instances data = loadData("cancer.txt");
        data.randomize(new java.util.Random(23));
        SplittedData splittedData = new SplittedData(data, 0.8);

        double[] degrees = {2, 3, 4};
        double[] gammas = {0.005, 0.05, 0.5};
        double[] c_i = {1, 0, -1, -2, -3, -4};
        double[] c_j = {3, 2, 1};
        double alpha = 1.5;

        double[][] kernelsMeasures = calcKernelsMeasures(splittedData, degrees, gammas);
        int bestKernelIndex = findBestKernelIndex(kernelsMeasures, alpha);
        double[][] cMeasures = calcCMeasures(splittedData, kernelsMeasures[bestKernelIndex], c_i, c_j);
        StringBuilder str = new StringBuilder();

        appendSectionKernel(str, kernelsMeasures, bestKernelIndex, alpha);
        appendSectionC(str, cMeasures);

        System.out.println(str.toString());
    }
}

class SplittedData {
    Instances train;
    Instances test;
    double trainProportion;

    public SplittedData(Instances dataFromFile, double p) throws Exception {
        RemovePercentage filter = new RemovePercentage();
        this.trainProportion = p;
        filter.setPercentage(p * 100);

        filter.setInvertSelection(false);
        filter.setInputFormat(dataFromFile);
        this.train = Filter.useFilter(dataFromFile, filter);

        filter.setInvertSelection(true);
        filter.setInputFormat(dataFromFile);
        this.test = Filter.useFilter(dataFromFile, filter);
    }
}
