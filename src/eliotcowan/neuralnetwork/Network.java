/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eliotcowan.neuralnetwork;

import static eliotcowan.neuralnetwork.NeuralNetwork.NUM_LAYERS;

/**
 *
 * @author elicowa
 */
public class Network {
    public static void feedForward(double[][][] weights, double[][] batch, int[] sizes, double[][][] outputs) { //layer X nueronto X nueronfrom
        for (int image = 0; image < batch.length; image++) { //feedforward
            outputs[image][0] = batch[image]; // Set input to image
            for (int layerTo = 1; layerTo < NUM_LAYERS; layerTo++) {
                outputs[image][layerTo] = propogateLayer(weights[layerTo - 1], outputs[image][layerTo - 1]);
            }
            //printLayer(outputs[image]);
        }
    }
    private static double[] propogateLayer(double[][] weights, double[] inputs) {
        double[] result = new double[weights.length]; //weights.length is the number of nuerons in layerTo
        for (int i = 0; i < weights.length; i++) { //per nueron
            result[i] = Util.sigmoid(Util.dotProduct(weights[i], inputs));
        }
        return result;
    }
}
