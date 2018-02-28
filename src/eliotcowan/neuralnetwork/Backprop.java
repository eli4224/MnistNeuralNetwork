/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eliotcowan.neuralnetwork;
//The problem: I am ony calculating the error for a single nueron rather than Etotal. I need calcualte how a weight effects a

import static eliotcowan.neuralnetwork.NeuralNetwork.LRATE;
import static eliotcowan.neuralnetwork.NeuralNetwork.NUM_LAYERS;
import static eliotcowan.neuralnetwork.NeuralNetwork.batchSize;
import static eliotcowan.neuralnetwork.NeuralNetwork.sizes;
import eliotcowan.neuralnetwork.Util.Derivatives;

/**
 *
 * @author elicowa
 */
public class Backprop {
    private static int[] realSizes;
    private static final double[][][] derE_derOut; //keep track of the derivitive of the total error with respect to the Out of a nueron.
    private static final double[][][] derOut_derNet; // keep track of the derivitive of the out with respect to the Net of a nueron. "der of sigmoid"
    private static final double[][][][] derE_derWeight; //track of the derivative of the total error with respect to a Weight. "The holy grail of backprop"
    private static final double[][][] derE_derNet; //derEROut * derOutRNet = "derE_derNet"
    static {
        derE_derOut = Util.build3dTemplate(batchSize, Util::buildNetworkTemplate);//Post Sigmoid
        derOut_derNet = Util.build3dTemplate(batchSize, Util::buildNetworkTemplate);//Pre Sigmoid
        derE_derNet = Util.build3dTemplate(batchSize, Util::buildNetworkTemplate);// Multiplication of prev two
        derE_derWeight = Util.build4dTemplate(batchSize, NeuralNetwork::getStartingWeights);
    }
    //propogation -forwardfeed
    public static void backPropigate(double[][][] outputs, int[] desiredOutputs, double[][][] weights) {
        //outputs: the array of all outputs for each iteration. image X layer X nueron
        //desiredOutputs: the array of all desired outputs. Image index w/ respect to weight and outputs X expected output.
        //weights: the array of all weights used in the working batch. Layer X nueronTo X nueronFrom
        //netoutputs: the array of all net values to each nueron. Image X Layer X nueron
        int batchSize = outputs.length;
        //gets starting errors.
        for (int batch = 0; batch < batchSize; batch++) {
            double[] oneHot = Util.oneHot(sizes[NUM_LAYERS - 1], desiredOutputs[batch]);
            for (int neuron = 0; neuron < sizes[NUM_LAYERS - 1]; neuron++) {
                derE_derOut[batch][NUM_LAYERS - 1][neuron] = Derivatives.derErrorROut(outputs[batch][NUM_LAYERS - 1][neuron], oneHot[neuron]);
            }
        }
        //starting backprop
        for (int image = 0; image < batchSize; image++) { //iteration number. "which image is being backproped"
            for (int layer = weights.length - 1; layer >= 0; layer--) { //weight layer number. "which weight layer is being proped"/ This part is tricky becuase the 0th layer of nueron connects to the 0th layer weight. So there can be a nth layer nuron but not an nth layer weight in the output layer.
                for (int to = 0; to < weights[layer].length; to++) { //nueron number. "which nueron's weights are being analysed"
                    derOut_derNet[image][layer + 1][to] = Derivatives.derOutRNet(outputs[image][layer + 1][to]); //see comment above on layer+1
                    derE_derNet[image][layer + 1][to] = derE_derOut[image][layer + 1][to] * derOut_derNet[image][layer + 1][to];
                    for (int from = 0; from < weights[layer][to].length; from++) {
                        derE_derWeight[image][layer][to][from] = derE_derNet[image][layer + 1][to] * Derivatives.derNetRWei(outputs[image][layer][from]);
                    }
                }
                //prep for next layer
                for (int from = 0; from < outputs[image][layer].length; from++) { //num of nuerons in previous layer
                    derE_derOut[image][layer][from] = Derivatives.derE_RderOut(derE_derNet[image][layer + 1], weights[layer], from);
                }
            }
        }
        updateWeights(derE_derWeight, weights);
    }
    public static double costFunc(int batchSize, double[][] outputs, double[] trueOutputs) { //calculation of cost
        //actual for each output unit - prediction
        double sum = 0;
        for (int i = 0; i < outputs[outputs.length - 2].length; i++) {
            sum += Math.pow((trueOutputs[i] - outputs[outputs.length - 2][i]), 2);
        }
        return sum * (1 / (2 * batchSize));
    }
    //weight update
    public static void updateWeights(double[][][][] derE_derWeight, double[][][] weights) { //previous Weights
        for (int lay = 0; lay < weights.length; lay++) {
            for (int toN = 0; toN < weights[lay].length; toN++) {
                for (int fromN = 0; fromN < weights[lay][toN].length; fromN++) {
                    double avg = 0;
                    for (int iterN = 0; iterN < derE_derWeight.length; iterN++) {
                        avg += derE_derWeight[iterN][lay][toN][fromN];
                    }
                    avg = avg / derE_derWeight.length;
                    weights[lay][toN][fromN] -= avg * LRATE;
                }
            }
        }
    }
}
//The problem: I am ony calculating the error for a single nueron rather than Etotal. I need calcualte how a weight effects a
//The problem: I am ony calculating the error for a single nueron rather than Etotal. I need calcualte how a weight effects a
//The problem: I am ony calculating the error for a single nueron rather than Etotal. I need calcualte how a weight effects a
