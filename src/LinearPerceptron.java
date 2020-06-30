import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

public class LinearPerceptron extends AbstractClassifier {

    private Instances instances;

    private double[] weights;

    private double learningRate = 1;
    private double adjustment = 0;

    private int numAttributes;
    private int bias = 0;
    private int stoppingCondition = 0;
    private int k = 4;

    private boolean weightsSet = false;
    private boolean training = true;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.instances = instances;
        this.numAttributes = instances.numAttributes();

        //set default value for weights if not specified
        if(!weightsSet){
            double[] newWeights = new double[instances.numAttributes()-1];
            for(int i = 0; i<instances.numAttributes()-1; i++){
                newWeights[i] = 1;
            }
            weights = newWeights;
        }

        getCapabilities().testWithFail(instances);

        innerBuild(instances);
        training=false;
    }

    private void innerBuild(Instances instances){
        int cont = 0;
        int x = 0;
        int runNum = 1;

        double t;
        double result;

        while (cont != instances.numInstances() && runNum != stoppingCondition) {
            Instance instance = instances.get(x);
            result = classifier(instance);
            if (instance.classValue() == 0) {
                t = -1;
            } else {
                t = 1;
            }
            if (result != t) {
                cont = 0;
                adjustment = (0.5 * (learningRate)) * (t - result);
                    // sends the weights double[] when set to Online update
                    weights(weights, adjustment, instance);
            } else {
                cont++;
            }

            if (x == instances.numInstances() - 1) {
                x = 0;
            } else {
                x++;
            }

            runNum++;
        }
        if (runNum == stoppingCondition) {
            System.out.println("Process cancelled as reached max number of iterations");
        }
    }

    //classify instance is called when classifying an instance
    public double classifyInstance(Instance instance) {
        return classifier(instance);
    }

    public double classifier(Instance instance){
        double result = 0;
        for (int i = 0; i < numAttributes - 1; i++) {
            result += weights[i] * instance.value(i);
        }
        if(training) {
            result = result + bias;
        }

        if (result < 0) {
            return -1;
        } else if (result > 0){
            return 1;
        } else {
            return 0;
        }
    }

    //adjust the weights for the perceptron
    public void weights(double[] arrayToChange, double adjustment, Instance instance) {
        for (int xj = 0; xj < numAttributes - 1; xj++) {
            arrayToChange[xj] = arrayToChange[xj] + (adjustment * instance.value(xj));
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.BINARY_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);

        return result;
    }


    // setters and getters
    // Some are unused, but kept so they may be called by marker if required for testing
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
        weightsSet = true;
    }

    public void setInstances(Instances instances) {
        this.instances = instances;
    }

    public void setBias(int bias) {
        this.bias = bias;
    }

    public void setStoppingCondition(int stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double[] getWeights() {
        return weights;
    }

    public Instances getInstances() {
        return instances;
    }

    public Instance getInstance (int i) {
        return instances.get(i);
    }

    public int getBias() {
        return bias;
    }

    public int getStoppingCondition() {
        return stoppingCondition;
    }
}
