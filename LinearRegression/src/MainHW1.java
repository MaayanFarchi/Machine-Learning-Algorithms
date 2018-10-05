import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.*;

public class MainHW1 {
	
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
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		// Load data:
		Instances trainingData = loadData("wind_training.txt");
		Instances testingData = loadData("wind_testing.txt");

		// Find best alpha and build classifier, considering all attributes:
		LinearRegression lr  = new LinearRegression();
		lr.buildClassifier(trainingData);
		double chosenAlpha = lr.getAlpha();
		double trainingError = lr.calculateMSE(trainingData);
		double testError = lr.calculateMSE(testingData);

		System.out.println(
                "The chosen alpha is: " + chosenAlpha +
                "\nTraining error with all features is: " + trainingError +
                "\nTest error with all features is: " + testError +
                "\n\nSeparating to sets of 3..."
        );

		// Build classifiers with all 3 attributes combinations:
        // errorList will hold all the combinations of sets of size 3
        // and the relevant MSE
        double[][] errorList = calculateBySets(trainingData, chosenAlpha);
        int minIndex = minIndex(errorList);

        // calculate the test error for the best 3 attributes
        setWeights(testingData, (int) errorList[minIndex][0], (int) errorList[minIndex][1],
                (int) errorList[minIndex][2]);
        setWeights(trainingData, (int) errorList[minIndex][0], (int) errorList[minIndex][1],
                (int) errorList[minIndex][2]);
        lr.buildClassifier(trainingData, chosenAlpha);
        double testErrorBest3 = lr.calculateMSE(testingData);

        // Print the results to the screen
        StringBuilder s = new StringBuilder("\n\tATTRIBUTES\t\t\t\tMSE\n");
        for (int i = 0; i < errorList.length; i++) {
            for (int j = 0; j < 3; j++)
                s.append(trainingData.attribute((int) errorList[i][j]).name() + " ");
            s.append("=> " + errorList[i][3] + "\n");
        }
        s.append("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++");
        s.append("\nBest 3 attributes are: ");
        for (int j = 0; j < 3; j++)
            s.append(trainingData.attribute((int) errorList[minIndex][j]).name() + " ");
        s.append("\nTraining error with these features is: " + errorList[minIndex][3] +
                 "\nTest error with these features is: " + testErrorBest3 +
                 "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++");
        System.out.print(s.toString());
	}

    /**
     * build linear classifiers considering all combinations of
     * attributes sets of size 3.
     * @param data
     * @param alpha
     * @return - an array of attributes and corresponding MSE
     * @throws Exception
     */
	private static double[][] calculateBySets(Instances data, double alpha) throws Exception {
        int attributes = data.numAttributes() - 1;
        double[][] errorList = new double[364][4]; // (14 choose 3) = 364
        int c = 0; // counter
        LinearRegression lr = new LinearRegression();

        for (int i = 0; i < attributes; i++) {
            for (int j = i+1; j < attributes; j++) {
                for (int k = j+1; k < attributes; k++) {
                    setWeights(data, i, j, k); // adjust the weights of the attributes
                    lr.buildClassifier(data, alpha); // considering only the 3 attributes (i,j,k)
                    // saving the 3 attributes and the MSE to the array
                    errorList[c][0] = i;    errorList[c][1] = j;    errorList[c][2] = k;
                    errorList[c++][3] = lr.calculateMSE(data);
                }
            }
        }

        return errorList;
    }

    /**
     * Setting the weights of all attributes other than i, j, k to 0
     * and i, j, k to 1. Thus ensuring only the 3 relevant attributes will
     * be considered in later calculations.
     * @param data
     * @param i
     * @param j
     * @param k
     */
    private static void setWeights(Instances data, int i, int j, int k) {
        for (int att = 0; att < data.numAttributes() && att != data.classIndex(); att++) {
            if (att != i && att != j && att != k)
                data.attribute(att).setWeight(0);
            else data.attribute(att).setWeight(1);
        }
    }

    /**
     * Finds the index of the minimal error among the sets of attributes
     * @param list
     * @return the index
     */
    private static int minIndex(double[][] list) {
	    int index = 0;
	    double val = list[0][3];
	    for (int i = 1; i < list.length; i++) {
	        if (list[i][3] < val) {
	            val = list[i][3];
	            index = i;
            }
        }
        return index;
    }
}
