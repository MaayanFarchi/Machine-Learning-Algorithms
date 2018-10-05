package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.expressionlanguage.common.MathFunctions;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.core.AttributeStats;

import java.util.Map;


class Node {
    Node[] children;
    Node parent;
    int attributeIndex;
    int attributeValue;
    double returnValue;
    int height;

    // constructor for the root
    public Node() {
        this.parent = null;
        this.attributeIndex = -1;
        this.attributeValue = -1;
        this.returnValue = -1;
        this.children = null;
        if (this.parent != null) this.height = this.parent.height + 1;
        else this.height = 0;
    }

    public Node(Node parent) {
        this.parent = parent;
        this.attributeIndex = -1;
        this.attributeValue = -1;
        this.returnValue = -1;
        this.children = null;
        if (this.parent != null) this.height = this.parent.height + 1;
        else this.height = 0;
    }
}

public class DecisionTree implements Classifier {
    Node rootNode;
    boolean toggleEntropy;
    double bestGain;
    double pValue = 1;

    @Override
    public void buildClassifier(Instances arg0) throws Exception {
        buildClassifier(arg0, 1, false);
    }

    public void buildClassifier(Instances arg0, double pValue) throws Exception {
        buildClassifier(arg0, pValue, false);
    }

    public void buildClassifier(Instances arg0, boolean toggleEntropy) throws Exception {
        buildClassifier(arg0, 1, toggleEntropy);
    }

    public void buildClassifier(Instances arg0, double pValue, boolean toggleEntropy) throws Exception {
        this.rootNode = new Node();
        this.toggleEntropy = toggleEntropy;
        this.pValue = pValue;
        buildTree(this.rootNode, arg0);
    }


    @Override
    public double classifyInstance(Instance instance) {
        // call the recursive auxiliary function for the tree-walk
        return classifyInstance(instance, this.rootNode).returnValue;
    }

    private Node classifyInstance(Instance instance, Node node) {
        // node has no further splits return the value of this node
        if (node.attributeIndex == -1) return node;

        int nextNodeIndex = -1;
        for (int i = 0; i < node.children.length; i++)
            if (instance.value(node.attributeIndex) == node.children[i].attributeValue) {
                nextNodeIndex = i;
                break;
            }
        // no matching value has been found, return the value of this node
        if (nextNodeIndex == -1) return node;
        // continue the tree-walk
        return classifyInstance(instance, node.children[nextNodeIndex]);
    }


    /**
     * Builds the decision tree on given data set using recursion.
     *
     * @param node
     * @param dataSet
     */
    public void buildTree(Node node, Instances dataSet) throws Exception {
        AttributeStats classStats = dataSet.attributeStats(dataSet.classIndex());
        double[] classWeights = classStats.nominalWeights;
        for (int i = 0; i < 2; i++) classWeights[i] = classWeights[i] / classStats.totalCount;
        // assign the return value of the node by the majority of the instances
        node.returnValue = Math.round(classWeights[1]);
        if (classWeights[0] == 1.0 || classWeights[0] == 0.0) return;

        int attributeIndex = bestDecisionAttributeIndex(dataSet);

        AttributeStats attributeStats = dataSet.attributeStats(attributeIndex);
        int allValues = dataSet.attribute(attributeIndex).numValues();
        Instances filteredData;

        if (bestGain == 0) return;
        double chiSquareStatistic = calcChiSquare(dataSet, attributeIndex, classWeights);
        int df = attributeStats.distinctCount - 1;
        if (this.pValue < 1 && df > 0 && !chiSquareDecision(chiSquareStatistic, this.pValue, df))
            return;

        node.attributeIndex = attributeIndex;
        node.children = new Node[attributeStats.distinctCount];

        for (int i = 0, c = 0; i < allValues; i++) {
            if (attributeStats.nominalCounts[i] == 0) continue;
            filteredData = filterData(dataSet, node.attributeIndex, i);
            node.children[c] = new Node(node);
            node.children[c].attributeValue = i;
            buildTree(node.children[c], filteredData);
            c++;
        }

    }

    /**
     * prints the tree according to the provided requirements.
     * this method uses an auxiliary for recursion printing
     */
    public void printTree() {
        StringBuilder tree = new StringBuilder("Root");
        printTree(this.rootNode, tree, 0);
        System.out.println(tree.toString());
    }

    private void printTree(Node node, StringBuilder tree, int level) {
        tree.append("\n");
        for (int i = 0; i < level; i++ ) tree.append("\t");
        if (node.attributeIndex == -1) {
            tree.append("\tLeaf. ");
            level--;
        }
        tree.append("Returning value: " + node.returnValue);
        if (node.attributeIndex == -1) return;

        level ++;
        for (int i = 0; i < node.children.length; i++) {
            tree.append("\n");
            for (int j = 0; j < level; j++ ) tree.append("\t");
            tree.append("If attribute " + node.attributeIndex + " = " + node.children[i].attributeValue);
            printTree(node.children[i], tree, level);
        }
    }

    /**
     * Calculate the index of the attribute that maximizes the Gain.
     *
     * @param dataSet - Instances to test
     * @return
     */
    private int bestDecisionAttributeIndex(Instances dataSet) throws Exception {
        int index = -1;
        double currGain;
        bestGain = -1;

        for (int i = 0; i < dataSet.numAttributes() - 1; i++) {
            currGain = calcGain(dataSet, i);
            if (currGain > bestGain) {
                bestGain = currGain;
                index = i;
            }
        }
        return index;
    }

    /**
     * Calculate the average error on a given instances set (could be the training, test or validation set).
     * The average error is the total number of classification mistakes on
     * the input instances set divided by the number of instances in the input set.
     *
     * @param dataSet
     * @return the average error
     */
    public double calcAvgError(Instances dataSet) {
        double prediction, actual;
        double mistakes = 0.0;
        int m = dataSet.numInstances();

        for (int i = 0; i < m; i++) {
            prediction = classifyInstance(dataSet.instance(i));
            actual = dataSet.instance(i).value(dataSet.classIndex());
            if (prediction != actual) mistakes++;
        }
        return mistakes / m;
    }

    /**
     * Calculate the average height and max height on a given instances set.
     *
     * @param dataSet - could be the training, test or validation set.
     * @return heightValue[0] = avgHeight, heightValue[1] = maxHeight
     */
    public double[] calcHeightValues(Instances dataSet) {
        int m = dataSet.numInstances();
        int instanceHeight;
        double maxHeight = 0.0;
        double sumOfHeights = 0.0;

        for (int i = 0; i < m; i++) {
            instanceHeight = classifyInstance(dataSet.instance(i), this.rootNode).height;
            if (instanceHeight > maxHeight) maxHeight = instanceHeight;
            sumOfHeights += instanceHeight;
        }

        double[] heightValues = {sumOfHeights / m, maxHeight};
        return heightValues;
    }

    /**
     * calculates the gain (giniGain or informationGain depending on the impurity measure)
     * of splitting the input data according to the attribute.
     *
     * @param trainingDataSubset - containing only instances relevant to this node
     * @param attributeIndex     - the index of the attribute currently tested
     * @return the gain measured by using the given attribute
     */
    public double calcGain(Instances trainingDataSubset, int attributeIndex) throws Exception {

        AttributeStats attributeStats = trainingDataSubset.attributeStats(attributeIndex);
        AttributeStats classStats = trainingDataSubset.attributeStats(trainingDataSubset.classIndex());
        AttributeStats valueClassStats;
        double[] attProportions = attributeStats.nominalWeights;
        double classProportion = classStats.nominalWeights[0] / classStats.totalCount;
        for (int i = 0; i < attProportions.length; i++)
            attProportions[i] = attProportions[i] / attributeStats.totalCount;
        Instances subset;

        // calculate the impurity measure of the father-set (S)
        double impurityFatherSet = (toggleEntropy) ? calcEntropy(classProportion) : calcGini(classProportion);

        // calculate the sum of impurity by filtering the trainingDataSubset by att values of attribute
        double sumOfChildrenImpurity = 0;

        for (int i = 0; i < attProportions.length; i++) {
            if (attProportions[i] == 0.0) continue;
            subset = filterData(trainingDataSubset, attributeIndex, i);
            valueClassStats = subset.attributeStats(subset.classIndex());
            classProportion = valueClassStats.nominalWeights[0] / valueClassStats.totalCount;
            // calculate the impurity measure of the child-sets (Sv)
            sumOfChildrenImpurity += attProportions[i] *
                    (toggleEntropy ? calcEntropy(classProportion) : calcGini(classProportion));
        }

        return impurityFatherSet - sumOfChildrenImpurity;
    }

    /**
     * Calculates the Entropy of a random variable.
     *
     * @param p - the proportion of the classified instances
     * @return entropy
     */
    public static double calcEntropy(double p) {
        if (p == 1 || p == 0) return 0;
        return -1 * (p * Math.log(p) / Math.log(2.0) + (1 - p) * Math.log(1 - p) / Math.log(2.0));
    }

    /**
     * Calculates the Gini of a random variable.
     *
     * @param p - the proportion of the classified instances
     * @return
     */
    public static double calcGini(double p) {
        return 1 - (p * p + (1 - p) * (1 - p));

    }

    /**
     * Filter the Instances by their value in a specific attribute
     *
     * @param data           - the Instances to filter
     * @param attributeIndex - the relevant attribute index
     * @param byValue        - the index of the attribute's value to be kept
     * @return - a filtered subset of the Instances
     */
    private static Instances filterData(Instances data, int attributeIndex, int byValue)
            throws Exception {
        RemoveWithValues filter = new RemoveWithValues();
        filter.setInvertSelection(true);
        // filtering indices starts from 1
        filter.setAttributeIndex("" + (attributeIndex + 1));
        filter.setNominalIndices("" + (byValue + 1));
        filter.setInputFormat(data);
        Instances out = Filter.useFilter(data, filter);
        filter = null;
        return out;
    }

    /**
     * Calculates the chi square statistic of splitting the
     * data according to the splitting attribute as learned in class.
     *
     * @param dataSet
     * @param attributeIndex
     * @return
     */
    private double calcChiSquare(Instances dataSet, int attributeIndex, double[] P) throws Exception {
        AttributeStats attributeStats = dataSet.attributeStats(attributeIndex);
        AttributeStats classStatsOfValue;
        Instances instancesWithValue;
        int[] D = attributeStats.nominalCounts;
        double[] E = new double[2];
        int[] N;
        double sum = 0;

        for (int i = 0; i < D.length; i++) {
            if (D[i] == 0) continue;
            instancesWithValue = filterData(dataSet, attributeIndex, i);
            classStatsOfValue = instancesWithValue.attributeStats(instancesWithValue.classIndex());
            N = classStatsOfValue.nominalCounts;

            for (int j = 0; j < 2; j++) {
                E[j] = D[i] * P[j];
                sum += (N[j] - E[j]) * (N[j] - E[j]) / E[j];
            }

        }
        return sum;
    }

    /**
     * @param statistic - provided by the #calcChiSquare()
     * @param pValue
     * @param df        - the degrees of freedom of the attribute values
     * @return true if the test proved informative, false otherwise
     */
    private boolean chiSquareDecision(double statistic, double pValue, int df) {
        double[][] chiSquareTable = {
                // pValue 0.75
                {0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438},
                // pValue 0.5
                {0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340},
                // pValue 0.25
                {1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845},
                // pValue 0.05
                {3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026},
                // pValue 0.005
                {7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300}
        };
        int pIndex;
        if (pValue == 0.75) pIndex = 0;
        else if (pValue == 0.5) pIndex = 1;
        else if (pValue == 0.25) pIndex = 2;
        else if (pValue == 0.05) pIndex = 3;
        else if (pValue == 0.005) pIndex = 4;
        else return true;

        double cutoff = chiSquareTable[pIndex][df - 1];
        return cutoff < statistic;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // Don't change
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // Don't change
        return null;
    }

}