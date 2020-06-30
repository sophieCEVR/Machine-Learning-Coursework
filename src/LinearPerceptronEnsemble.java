import weka.core.Instances;
import weka.core.Instance;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class LinearPerceptronEnsemble {

    int ensembleSize = 50;
    double attributeProportion = 0.5;
    float totalVotes = 0;

    LinearPerceptron[] linearPerceptrons = new LinearPerceptron[ensembleSize];
    int[][] attributesRemoved;
    Classification[] classifications = new Classification[]{};

    public void buildClassifier(Instances instances, double attributeProportion) throws Exception {
        this.attributeProportion = attributeProportion;

        for (int i = 0; i < ensembleSize; i++) {
            linearPerceptrons[i] = new LinearPerceptron();
            linearPerceptrons[i].setInstances(new Instances(instances));
            Collections.shuffle(linearPerceptrons[i].getInstances());
        }

        int y = (int) ((instances.numAttributes() - 1) * attributeProportion); //the number of attributes in each split
        attributesRemoved = new int[ensembleSize][y];
        int[] toMove = new int[y];

        for (int c = 0; c < ensembleSize; c++) {

            int[] available = new int[instances.numAttributes() - 1];
            for (int i = 0; i < instances.numAttributes() - 1;i++) {
                available[i] = i;
            }

            for (int i = 0; i < y; i++) {
                Random rand = new Random(); //instance of random class
                int random = rand.nextInt(available.length);
                toMove[i] = available[random];

                int[] temp = new int[available.length-1];
                for (int j = 0; j < available.length-1;j++) {
                    if (available[j]!=available[random]) {
                        temp[j]= available[j];
                    }
                }
                available = new int[temp.length];
                System.arraycopy(temp, 0, available, 0, temp.length);
            }

            //delete the unwanted attributes from each set of instances in the Ensemble
            Arrays.sort(toMove);
            for (int k = toMove.length-1; k > -1; k--) {
                linearPerceptrons[c].getInstances().deleteAttributeAt(toMove[k]);
            }

            //Save the deleted attributes to recreate later.
            System.arraycopy(toMove, 0, attributesRemoved[c], 0, y);
        }

        //Build a classifier for each Instances object (each instances with the attributes used
        for (int i=0; i<ensembleSize; i++){
            linearPerceptrons[i].buildClassifier(linearPerceptrons[i].getInstances());
        }

        // DistributionForInstance calls classifyInstance and returns the classification object type
        Instance instance = linearPerceptrons[0].getInstance(0);
        classifications = distributionForInstance(instance);

    }

    class Classification {
        double classification;
        int count = 0;
        double voteProportion = 0.0;
    }

    public double classifyInstance(Instance instance){
        classifications = new Classification[linearPerceptrons[0].getInstance(0).classAttribute().numValues()];

        for(int i=0; i < classifications.length; i++){
            classifications[i] = new Classification();
            classifications[i].classification = Double.parseDouble(linearPerceptrons[0].getInstance(0).classAttribute(). value(i));
        }

        for(int i =0; i<ensembleSize;i++) {
            double currentClassification = (linearPerceptrons[i].classifyInstance(instance));
            for (int j=0; j< classifications.length; j++) {
               if (classifications[j].classification == currentClassification){
                   classifications[j].count++;
                   totalVotes ++;
               }
           }
        }

        Classification classification = new Classification();
        for (int j=0; j < classifications.length; j++) {
            if(classifications[j].count > classification.count){
                classification = classifications[j];
            }
        }
        return classification.classification;
    }



    public Classification[] distributionForInstance(Instance instance){
        classifyInstance(instance);
        for (Classification classification : classifications) {
            if(classification.count != 0) {
                classification.voteProportion = totalVotes / classification.count;
            }
        }
        return classifications;
    }
}