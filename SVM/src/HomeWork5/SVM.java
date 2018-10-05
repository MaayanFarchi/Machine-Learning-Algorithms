package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instance;
import weka.core.Instances;

public class SVM {
    public SMO m_smo;

    public SVM() {
        this.m_smo = new SMO();
    }

    public void buildClassifier(Instances instances) throws Exception {
        m_smo.buildClassifier(instances);
    }

    public int[] calcConfusion(Instances instances) throws Exception {
        // [TP, FP, TN, FN]
        int[] confusionMatrix = new int[4];
        double classification;
        int loc;

        for (Instance instance : instances) {
            classification = this.m_smo.classifyInstance(instance);
            // True classification
            if (instance.classValue() == classification)
                loc = 2 * (1 - (int) classification);
            // False classification
            else loc = (classification == 0) ? 3 : 1;
            confusionMatrix[loc]++;
        }
        return confusionMatrix;
    }

    public void setKernel(Kernel kernel) {
        this.m_smo.setKernel(kernel);
    }

    public void setC(double v) {
        this.m_smo.setC(v);
    }

    public double getC() {
        return this.m_smo.getC();
    }
}
