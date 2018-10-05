package HomeWork2;

import java.io.*;
import java.util.Arrays;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.core.AttributeStats;

public class MainHW2 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    /**
     * Sets the class index as the last attribute.
     *
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {
        Instances trainingCancer = loadData("cancer_train.txt");
        Instances testingCancer = loadData("cancer_test.txt");
        Instances validationCancer = loadData("cancer_validation.txt");

        double[] pValues = {1, 0.75, 0.5, 0.25, 0.05, 0.005};
        DecisionTree tree = new DecisionTree();
        StringBuilder results = new StringBuilder("");

        tree.buildClassifier(trainingCancer, true);
        double validationErrorEntropy = tree.calcAvgError(validationCancer);
        results.append("Validation error using Entropy: " + validationErrorEntropy + "\n");

        tree.buildClassifier(trainingCancer);
        double validationErrorGini = tree.calcAvgError(validationCancer);
        results.append("Validation error using Gini: " + validationErrorGini + "\n");

        results.append("----------------------------------------------------\n");

        double trainingError, validationError, testError, bestP = 1;
        double bestValidationError = Double.MAX_VALUE;
        double[] heightValues;
        for (double p : pValues) {
            tree.buildClassifier(trainingCancer, p);
            trainingError = tree.calcAvgError(trainingCancer);
            heightValues = tree.calcHeightValues(validationCancer);
            validationError = tree.calcAvgError(validationCancer);
            if (validationError < bestValidationError) {
                bestValidationError = validationError;
                bestP = p;
            }
            results.append("Decision Tree with p_value of: " + p +
                    "\nThe train error of the decision tree is " + trainingError +
                    "\nMax height on validation data: " + (int) heightValues[1] +
                    "\nAverage height on validation data: " + heightValues[0] +
                    "\nThe validation error of the decision tree is " + validationError +
                    "\n----------------------------------------------------\n");
        }

        tree.buildClassifier(trainingCancer, bestP);
        testError = tree.calcAvgError(testingCancer);
        results.append("Best validation error at p_value = " + bestP +
                "\nTest error with best tree: " + testError +
                "\nRepresentation of the best tree by ‘if statements’:\n\n");
        System.out.print(results.toString());
        tree.printTree();

    }
}
