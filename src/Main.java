
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws Exception {

        String trainArff = "training-data/part1.arff";
        String testArff = "training-data/part1.arff";

        Instances train = loadData(trainArff);
        Instances test = loadData(testArff);

        train.setClassIndex(train.numAttributes()-1);
        test.setClassIndex(test.numAttributes()-1);

        LinearPerceptron linearPerceptron = new LinearPerceptron();

        linearPerceptron.buildClassifier(train);

        System.out.println("Linear Perceptron Weights");
        System.out.println(Arrays.toString(linearPerceptron.getWeights()));
        System.out.println("Linear Perceptron Test Data Classifications");
        for (Instance instance : test) {
            System.out.println(linearPerceptron.classifyInstance(instance));
        }

        EnhancedLinearPerceptron enhanced = new EnhancedLinearPerceptron();
        enhanced.setStandardise(true);
        enhanced.setK(10);
        enhanced.buildClassifier(train);
        System.out.println("\nEnhanced Linear Perceptron Weights");
        System.out.println(Arrays.toString(enhanced.getWeights()));
        System.out.println("Enhanced Linear Perceptron Test Data Classifications");
        for (Instance instance : test) {
            System.out.println(enhanced.classifyInstance(instance));
        }

        String ensembleArff = "training-data/part2.arff";
        Instances ensembleinstances = loadData(ensembleArff);
        ensembleinstances.setClassIndex(ensembleinstances.numAttributes()-1);

        LinearPerceptronEnsemble ensemble = new LinearPerceptronEnsemble();
        ensemble.buildClassifier(ensembleinstances, 0.5);

        System.out.println("\nLinear Perceptron Ensemble Weights");
        for(LinearPerceptron perceptron: ensemble.linearPerceptrons){
            System.out.println(Arrays.toString(perceptron.getWeights()));
        }

        System.out.println("Linear Perceptron Ensemble Test Data Classifications");
        for (Instance instance : test) {
            System.out.println(ensemble.classifyInstance(instance));
        }

    }


    public static Instances loadData(String filePath){
        Instances instances = null;

        try{
            FileReader reader = new FileReader(filePath);
            instances = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return instances;
    }
}

