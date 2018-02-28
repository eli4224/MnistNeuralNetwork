/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eliotcowan.neuralnetwork;

import java.util.function.Supplier;

/**
 *
 * @author elicowa
 */
public class Util {
    public static double[] oneHot(int size, int hot) {
        double[] result = new double[size];
        result[hot] = 1;
        return result;
    }
    public static void printLayer(double[][] outputs, int label) {
        double[] oneHot = Util.oneHot(NeuralNetwork.sizes[NeuralNetwork.NUM_LAYERS - 1], label);
        for (int j = 0; j < outputs[NeuralNetwork.NUM_LAYERS - 1].length; j++) {
            System.out.printf("%d:%1.1f,", (int) oneHot[j], outputs[NeuralNetwork.NUM_LAYERS - 1][j]);
        }
        System.out.println("");
    }
    public static double sigmoid(double a) {
        return (1d / (1d + java.lang.Math.exp(-a)));
    }
    public static double dotProduct(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    public static double[][][] build3dTemplate(int size, Supplier<double[][]> base) {
        double[][][] result = new double[size][][];
        for (int i = 0; i < size; i++) {
            result[i] = base.get();
        }
        return result;
    }
    public static double[][][][] build4dTemplate(int size, Supplier<double[][][]> base) {
        double[][][][] result = new double[size][][][];
        for (int i = 0; i < size; i++) {
            result[i] = base.get();
        }
        return result;
    }
    public static double[][] buildNetworkTemplate() {
        double[][] result = new double[NeuralNetwork.NUM_LAYERS][]; //due to removal of softLayer
        for (int k = 0; k < result.length; k++) {
            result[k] = new double[NeuralNetwork.sizes[k]];
        }
        return result;
    }
    public static class Derivatives {
        public static double derOutRNet(double out) { //der of sigmoid function
            return out * (1 - out);
        }
        public static double derErrorROut(double actual, double expected) {
            return actual - expected;
        }
        public static double derNetRWei(double out) {
            return out;
        }
        public static double derE_RderOut(double[] nextLayerNets, double[][] weights, int fromIndex) {
            int sum = 0;
            for (int i = 0; i < nextLayerNets.length; i++) {
                sum += nextLayerNets[i] * weights[i][fromIndex];
            }
            return sum;
        }
    }
    public static int rand(int cap) {
        return (int) (cap * Math.random());
    }
}
