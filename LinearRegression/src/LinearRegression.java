import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes()-1;
		m_coefficients = new double[m_truNumAttributes + 1];
        findAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);
	}


	public void buildClassifier(Instances trainingData, double alpha) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes()-1;
		m_coefficients = new double[m_truNumAttributes + 1];
		m_alpha = alpha;
		m_coefficients = gradientDescent(trainingData);
	}

    public double getAlpha() {
	    return this.m_alpha;
    }

	private void resetTheta() {
	    int m = m_coefficients.length;
	    for (int i = 0; i < m; i++) m_coefficients[i] = 1;
    }

    private void updateTheta(double alpha, Instances data) throws Exception {
        int n = m_coefficients.length;
        int m = data.numInstances();
        double x_t, derivative_t;
        Instance instance;
	    double[] temp_coefficients = new double[n];

        for (int t = 0; t < n; t++) {
            // skip attributes with weight 0
            if (t > 0 && data.attribute(t-1).weight() == 0) continue;
            derivative_t = 0;
            for (int i = 0; i < m; i++) {
                instance = data.instance(i);
                x_t = (t == 0) ? 1 : instance.value(t-1);
                derivative_t += (regressionPrediction(instance) - instance.value(m_ClassIndex)) * x_t;
            }
            temp_coefficients[t] = m_coefficients[t] - alpha * derivative_t / m;
        }

        for (int t = 0; t < n; t++) {
            m_coefficients[t] = temp_coefficients[t];
        }
    }
	
	private void findAlpha(Instances data) throws Exception {
		double alpha, pre_MSE, curr_MSE;
        double best_MSE = Double.MAX_VALUE;

		for (int i = 0; i > -18; i--) {
		    alpha = Math.pow(3, i);
		    this.resetTheta();
            pre_MSE = Double.MAX_VALUE;
		    for (int j = 1; j <= 20000; j++) {
                this.updateTheta(alpha, data);
		        if (j % 100 == 0) {
                    curr_MSE = calculateMSE(data);
                    if (curr_MSE > pre_MSE) break;
                    pre_MSE = curr_MSE;
                }
            }

            if (pre_MSE < best_MSE) {
		        best_MSE = pre_MSE;
                m_alpha = alpha;
            }
        }
    }
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
	    double pre_MSE = Double.MAX_VALUE;
	    double curr_MSE, dif_ERR;
        double epsilon = 0.003;
        this.resetTheta();

	    do {
            for (int i = 0; i < 100; i++)
                this.updateTheta(m_alpha, trainingData);
            curr_MSE = calculateMSE(trainingData);
            dif_ERR = pre_MSE - curr_MSE;
            pre_MSE = curr_MSE;
        } while (Math.abs(dif_ERR) > epsilon);

	    return m_coefficients;
    }
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
	    double predictedY = m_coefficients[0];
	    for (int i = 0; i < m_truNumAttributes && i != m_ClassIndex; i++)
	        // skip attributes with weight 0
	        if (instance.attribute(i).weight() != 0)
	            predictedY += m_coefficients[i+1]*instance.value(i);

	    return predictedY;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
	    int m = data.numInstances();
	    double iE;
	    double SE = 0;

	    for (int i = 0; i < m; i++) {
	        iE = regressionPrediction(data.instance(i)) - data.instance(i).value(m_ClassIndex);
	        SE += iE*iE;
        }

		return SE / (2*m);
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
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
