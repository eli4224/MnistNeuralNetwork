/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eliotcowan.neuralnetwork;

import averycowan.util.Tensor;
import java.io.IOException;
import java.util.Random;

/**
 *
 * @author elicowa
 */
public class NeuralNetwork {
    public static final int batchSize = 1;
    public static final int sizes[] = {784, 400, 200, 100, 10};
    public static final double LRATE = .03;//0.00004d; // Learning Rate
    public static final int NUM_LAYERS = sizes.length;
    private static final int TRAINING_TIME = 60000;
    private static final int TESTING_TIME = 1000;
    private static final int DEBUG_LEVEL = TRAINING_TIME / 100;
    public static void main(String[] args) {
        // TODO code application logic here
        try {
            System.out.println("Reading labels");
            MnistReader.readlabels();
            System.out.println("Reading images");
            MnistReader.readimages();
        } catch (IOException e) {
            throw new RuntimeException("Input images or labels not found", e);
        }
        System.out.println("Running Network");
        double[][][] weights = getStartingWeights();
        double[][][] outputs = Util.build3dTemplate(batchSize, Util::buildNetworkTemplate);
        int totalCorrect = 0;
        int numRands = 10000;
        int[] rand = new int[numRands];
        for (int i = 0; i < numRands; i++) {
            rand[i] = (int) (Math.random() * (double) MnistReader.labels.length);
        }
        for (int ite = 0; ite < TRAINING_TIME + TESTING_TIME; ite++) {
            double[][] batch;
            int[] labels;
            if (ite == TRAINING_TIME) {
                totalCorrect = 0;
            }
            {
                batch = new double[batchSize][];
                labels = new int[batchSize];
                for (int image = 0; image < batchSize; image++) {
                    int imagenum = Util.rand(MnistReader.labels.length);
                    batch[image] = MnistReader.data[imagenum]; //TODO don't reuse images
                    labels[image] = MnistReader.labels[imagenum];
                }
            }
            Network.feedForward(weights, batch, sizes, outputs);
            totalCorrect = updateAccuracy(outputs, labels, ite, totalCorrect);
            Backprop.backPropigate(outputs, labels, weights);
        }
        System.out.printf("You got %d correct out of 1000!\n", totalCorrect);
        //for (int m = 0; m < batchSize; m++) {
        //for (int image = 0; image < outputs[m].length; image++) {
        //System.out.print(outputs[m][image] + ",");
        //}
        //System.out.println(MnistReader.labels[m]);
        //}
    }
    public static int updateAccuracy(double[][][] outputs, int[] labels, int ite, int totalCorrect) {
        int correctClassifications = 0;
        for (int iter = 0; iter < outputs.length; iter++) {
            if (Tensor.greatest(outputs[iter][NUM_LAYERS - 1]) == labels[iter]) {//Compare to the index of the biggest thing in the last real layer
                correctClassifications++;
            }
        }
        if (ite % DEBUG_LEVEL == 0) {
            Util.printLayer(outputs[0], labels[0]);
            System.out.printf("Ite = %d, Correct = %d\n", ite, totalCorrect);
            System.out.println(correctClassifications > 0 ? "YAY" : "RIPURFACE");
        }
        return totalCorrect + correctClassifications;
    }
//MnistReader.getEpoche(batchSize);
//}
    private static double[][][] getStartingWeights(int[] size) {
        //create array
        double[][][] result = new double[NUM_LAYERS - 1][][]; // layer x output from nueron
        for (int i = 0; i < result.length; i++) {
            result[i] = new double[size[i + 1]][size[i]];
        }
        //create weights
        Random r = new Random(); //For nextGaussian()
        for (int i = 0; i < NUM_LAYERS - 1; i++) { //-1 due to weight layers, -1 due to soft layer
            for (int k = 0; k < size[i + 1]; k++) {
                for (int j = 0; j < size[i]; j++) {
                    result[i][k][j] = (r.nextGaussian()) / 100;
                }
            }
        }
        return result;
    }
    static double[][][] getStartingWeights() {
        return getStartingWeights(sizes);
    }
}
