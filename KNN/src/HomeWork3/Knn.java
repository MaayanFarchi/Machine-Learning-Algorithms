package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import java.util.*;

class DistanceCalculator {

    private double p;
    private boolean calcEfficient;
    private double cutOfValue;

    public DistanceCalculator(double p, Knn.DistanceCheck distanceCheck) {
        this.p = p;
        this.calcEfficient = distanceCheck == Knn.DistanceCheck.Efficient;
        this.cutOfValue = Double.MAX_VALUE;
    }

    public void setCutOfValue(double cutOfValue) {
        this.cutOfValue = cutOfValue;
    }

    /**
     * We leave it up to you wheter you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it has a class variables.
     */
    public double distance(Instance one, Instance two) {
        if (p == Double.MAX_VALUE) return lInfinityDistance(one, two);
        else return lpDistance(one, two);
    }

    /**
     * Returns the Lp distance between 2 instances.
     *
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two) {
        int n = one.numAttributes() - 1;
        double sum = 0.0;
        double poweredCutOfValue = 1.0;
        if (this.calcEfficient) poweredCutOfValue = Math.pow(this.cutOfValue, p);

        for (int i = 0; i < n; i++) {
            sum += Math.pow(Math.abs(one.value(i) - two.value(i)), this.p);
            if (this.calcEfficient && sum >= poweredCutOfValue) return Double.MAX_VALUE;
        }
        return Math.pow(sum, 1 / this.p);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     *
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        int n = one.numAttributes() - 1;
        double absMax = 0.0;
        double current;

        for (int i = 0; i < n; i++) {
            current = Math.abs(one.value(i) - two.value(i));
            if (current > absMax) {
                absMax = current;
                if (this.calcEfficient && absMax >= this.cutOfValue) return Double.MAX_VALUE;
            }
        }
        return absMax;
    }

    /*
    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     *//*
    private double efficientLpDistance(Instance one, Instance two) {
        int n = one.numAttributes() - 1;
        double sum = 0.0;

        for (int i = 0; i < n; i++) {
            sum += Math.pow(Math.abs(one.value(i) - two.value(i)), this.p);
            if (sum >= this.cutOfValue) return Double.MAX_VALUE;
        }
        return Math.pow(sum, 1 / this.p);
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     *//*
    private double efficientLInfinityDistance(Instance one, Instance two) {
        return 0.0;
    }
    */
}

class NeighborComparator implements Comparator<Neighbor> {

    @Override
    public int compare(Neighbor a, Neighbor b) {
        return b.distance.compareTo(a.distance);
    }
}

class Neighbor {
    Instance instance = null;
    Double distance = 0.0;

    public Neighbor (Instance instance, Double distance) {
        this.instance = instance;
        this.distance = distance;
    }

    public void setDistance(Double distance) {
        this.distance = distance;
    }

    public void setInstance(Instance instance) {
        this.instance = instance;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck {Regular, Efficient}

    public enum PredictionType {Uniform, Weighted}

    private Instances m_trainingInstances;
    private int k;
    private double p;
    private DistanceCheck distanceCheck;
    private PredictionType predictionType;
    private boolean measureTime;
    private long predictionRunTime;

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        this.m_trainingInstances = instances;
        this.predictionType = PredictionType.Uniform;
        this.distanceCheck = DistanceCheck.Regular;
        this.measureTime = false;
    }

    public void setK(int k) {
        this.k = k;
    }

    public void setP(double p) {
        this.p = p;
    }

    public void setDistanceCheck(DistanceCheck distanceCheck) {
        this.distanceCheck = distanceCheck;
    }

    public void setPredictionType(PredictionType predictionType) {
        this.predictionType = predictionType;
    }

    public void setMeasureTime(boolean measureTime) {
        this.measureTime = measureTime;
    }

    public long getPredictionRunTime() {
        return predictionRunTime;
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        double prediction;
        PriorityQueue<Neighbor> kNN = findNearestNeighbors(instance);
        switch (this.predictionType){
            case Weighted:
                prediction = getWeightedAverageValue(kNN);
                break;
            default:
                prediction = getAverageValue(kNN);
                break;
        }

        return prediction;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     *
     * @param instances
     * @return
     */
    public double calcAvgError(Instances instances) {
        double diff, prediction;
        double sumOfAbsoluteErrors = 0.0;


        for (Instance instance : instances) {
            prediction = regressionPrediction(instance);
            diff = instance.value(instance.classIndex()) - prediction;
            sumOfAbsoluteErrors += Math.abs(diff);
        }

        return sumOfAbsoluteErrors / instances.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param instances instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) throws Exception {
        StratifiedRemoveFolds foldsFilter = new StratifiedRemoveFolds();
        foldsFilter.setNumFolds(num_of_folds);
        Instances initialClassMember = this.m_trainingInstances;
        Instances validationFold;
        double sumOfFoldsErrors = 0.0;
        long predictionClock = 0;
        if (this.measureTime) this.predictionRunTime = 0;

        for (int i = 1; i <= num_of_folds; i++) {
            foldsFilter.setFold(i); // focus on fold 'i'

            // set the rest of the data ("training data")
            foldsFilter.setInvertSelection(true);
            foldsFilter.setInputFormat(instances);
            this.m_trainingInstances = Filter.useFilter(instances, foldsFilter);

            // set the validation data part
            foldsFilter.setInvertSelection(false);
            foldsFilter.setInputFormat(instances);
            validationFold = Filter.useFilter(instances, foldsFilter);

            // Calculate the error of the fold, considering the rest of the data
            // measure time if requested
            if (this.measureTime) predictionClock = System.nanoTime();
            sumOfFoldsErrors += calcAvgError(validationFold);
            if (this.measureTime) {
                predictionClock = System.nanoTime() - predictionClock;
                this.predictionRunTime += predictionClock;
            }
        }

        this.m_trainingInstances = initialClassMember; // return the class member to initial value

        return sumOfFoldsErrors / num_of_folds;
    }


    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public PriorityQueue<Neighbor> findNearestNeighbors(Instance instance) {
        PriorityQueue<Neighbor> kNN = new PriorityQueue<>(new NeighborComparator());

        DistanceCalculator dc = new DistanceCalculator
                (this.p, this.distanceCheck);

        double distance;
        Neighbor furthest, neighbor;


        // adds all first k instances to the mapping
        for (int i = 0; i < k; i++) {
            distance = dc.distance(instance, this.m_trainingInstances.instance(i));
            neighbor = new Neighbor(this.m_trainingInstances.instance(i), distance);
            kNN.add(neighbor);
        }

        furthest = kNN.peek();
        dc.setCutOfValue(furthest.distance);

        // replace the furthest neighbor if necessary
        for (int i = k; i < this.m_trainingInstances.numInstances(); i++) {
            distance = dc.distance(instance, this.m_trainingInstances.instance(i));

            if (distance < furthest.distance) {
                kNN.remove(furthest);
                neighbor = new Neighbor(this.m_trainingInstances.instance(i), distance);
                kNN.add(neighbor);
                furthest = kNN.peek();
                dc.setCutOfValue(furthest.distance);
            }
        }

        return kNN;
    }


    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param
     * @return
     */
    public double getAverageValue(PriorityQueue<Neighbor> neighbors) {
        double sum = 0.0;
        for (Neighbor neighbor : neighbors)
            sum += neighbor.instance.value(neighbor.instance.classIndex());

        return sum / neighbors.size();
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @return
     */
    public double getWeightedAverageValue(PriorityQueue<Neighbor> neighbors) {
        double weight, value, distance;
        double sumOfWeights = 0.0;
        double sumOfWeightedValues = 0.0;

        for (Neighbor neighbor : neighbors) {
            distance = neighbor.distance;
            value = neighbor.instance.value(neighbor.instance.classIndex());
            if (distance == 0) return value;
            weight = 1 / Math.pow(distance, 2);

            sumOfWeights += weight;
            sumOfWeightedValues += weight * value;
        }

        return sumOfWeightedValues / sumOfWeights;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
