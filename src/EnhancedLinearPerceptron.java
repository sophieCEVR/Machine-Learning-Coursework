import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.ThreadLocalRandom;

public class EnhancedLinearPerceptron extends AbstractClassifier {

    private Instances instances;

    private double[] weights;
    private double[] meanPerAttribute;
    private double[] STDPerAttribute;
    private double[] offline;

    private double learningRate = 1;
    private double adjustment = 0;

    private int numAttributes;
    private int bias = 0;
    private int stoppingCondition = 0;
    private int k = 4;

    private boolean standardise = false;
    private boolean online = true;
    private boolean crossValidate = false;
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

        if (standardise) {standardiseInstances(instances);}
        if (crossValidate){ kFoldCrossValidate(); }
        if (!online){ offline = new double[instances.numAttributes()-1]; }

        innerBuild(instances);
        training = false;
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
                if(online) {
                    // sends the weights double[] when set to Online update
                    weights(weights, adjustment, instance);
                } else {
                    // sends the offline double[] when set to Online update
                    weights(offline, adjustment, instance);
                }
            } else {
                cont++;
            }

            if (x == instances.numInstances() - 1) {
                if(!online){
                    for (int xj = 0; xj < numAttributes - 1; xj++) {
                        weights[xj] = weights[xj] + offline[xj];
                        offline[xj] = 0;
                    }
                }
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
        if (standardise) {
            standardiseInstance(instance);
        }
        return classifier(instance);
    }

    private double classifier(Instance instance){
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
    private void weights(double[] arrayToChange, double adjustment, Instance instance) {
        for (int xj = 0; xj < numAttributes - 1; xj++) {
            arrayToChange[xj] = arrayToChange[xj] + (adjustment * instance.value(xj));
        }
    }

    private void standardiseInstances(Instances instances) {

        meanPerAttribute = new double[instances.numAttributes() - 1];
        STDPerAttribute = new double[instances.numAttributes() - 1];

        for (int attribute = 0; attribute < instances.numAttributes() - 1; attribute++) {
            // get the mean of all attributes in the list of instances
            meanPerAttribute[attribute] = instances.meanOrMode(attribute);

            //get the standard deviation of all attributes in the list of instances
            STDPerAttribute[attribute] = Math.sqrt(instances.variance(attribute));
        }

        // work out standardised version of each instance
        for (Instance instance : instances) {
            standardiseInstance(instance);
        }

    }

    private void standardiseInstance(Instance instance) {
        for (int attribute = 0; attribute < instance.numAttributes()-1; attribute++) {
            double standardised = (instance.value(attribute) - meanPerAttribute[attribute]) / STDPerAttribute[attribute];
            instance.setValue(attribute, standardised);
        }
    }

    private void resetStandardise(Instances instances){
        meanPerAttribute = new double[instances.numAttributes() - 1];
        STDPerAttribute = new double[instances.numAttributes() - 1];
    }

    private void kFoldCrossValidate() {
        Instances CVInstances = new Instances(instances);
        Collections.shuffle(CVInstances);

        //define the folds
        int[][] folds = new int[k][instances.numInstances()/k];
        int current = 0;

        for (int i = 0; i < k; i++) {
            for (int y =0; y < CVInstances.numInstances()/k; y++) {
                folds[i][y] = current;
                current++;
            }
        }

        //Create the models
        Instances[] onlineModel = createModel(CVInstances, folds);
        Instances[] offlineModel = createModel(CVInstances, folds);


        //Test the models
        double onlineAccuracy = testModel(true,onlineModel);
        offline = new double[offlineModel[0].numAttributes()-1];
        double offlineAccuracy = testModel(false,offlineModel);

        //Chose the update method based on the accuracy calculated
        if(onlineAccuracy >= offlineAccuracy) {
            online = true;
        }
        else {
            online = false;
        }

    }


    private double accuracy(Instances test) {
        double correct = 0;

        for (Instance instance : test){
            double answer = instance.classValue();
            if (classifyInstance(instance) == answer){
                correct++;
            }
        }
        return (correct/test.numInstances())*100;
    }


    private double testModel(boolean updateMethod, Instances[] instancesArr) {
        online = updateMethod;
        innerBuild(instancesArr[0]);
        standardiseInstances(instancesArr[0]);
        double accuracy = accuracy(instancesArr[1]);
        resetStandardise(instancesArr[0]);
        return accuracy;
    }

    private Instances[] createModel(Instances CVInstances, int[][] folds){
        Instances trainInstances = new Instances(CVInstances);
        Instances testInstances = new Instances(instances,0);

        int randomNum = ThreadLocalRandom.current().nextInt(0, k-1 );

        //separate the folds into train and test instances groups
        for (int i = 0; i < k; i++) {
            if(i==randomNum) {
                for (int y = 0; y < (instances.numInstances() / k); y++) {
                    testInstances.add(trainInstances.get(folds[i][y]));
                    trainInstances.remove(folds[i][y]);
                }
            }

        }
        return new Instances[]{trainInstances, testInstances};
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

    public void setK(int k){this.k = k;}

    public void setStoppingCondition(int stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
    }

    public void setStandardise(boolean standardise) {
        this.standardise = standardise;
    }

    public void setCrossValidate(boolean crossValidate) {
        this.crossValidate = crossValidate;
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

    public boolean getStandardise() {
        return standardise;
    }

    public boolean getCrossValidate() {
        return crossValidate;
    }

    public int getBias() {
        return bias;
    }

    public int getStoppingCondition() {
        return stoppingCondition;
    }


}
