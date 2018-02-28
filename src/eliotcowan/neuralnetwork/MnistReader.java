/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eliotcowan.neuralnetwork;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 *
 * @author elicowa
 */
public class MnistReader {
    //Vars
    public static byte[] labels;
    public static byte[][] byteData;
    public static double[][] data;
    //Methods
    public static void readlabels() throws IOException {
        FileInputStream i = new FileInputStream(new File("/Users/elicowa/Downloads/train-labels-idx1-ubyte"));
        DataInputStream in = new DataInputStream(i);
        int magic = in.readInt(); //header, to be ignored
        int images = in.readInt(); //number of images
        labels = new byte[images];
        in.readFully(labels); //pastes the rest of the datastream into labels
    }
    public static void readimages() throws IOException {
        FileInputStream i = null;
        try {
            i = new FileInputStream(new File("/Users/elicowa/Downloads/train-images-idx3-ubyte"));
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Input images not found", e);
        }
        DataInputStream in = new DataInputStream(i);
        int magic = in.readInt(); //file header, to be ignored
        int images = in.readInt(); //number of images
        int rows = in.readInt(); //number of rows
        int column = in.readInt(); //number of cols
        byteData = new byte[images][rows * column]; //number of images by the number of pixels per image (#pixal rows * #pixal cosl)
        for (int im = 0; im < images; im++) {
            in.readFully(byteData[im]); //pastes the rest of the data stream into data
            //System.out.println(data[im][0].getClass());
        }
        data = new double[images][rows * column];
        for (int k = 0; k < images; k++) {
            for (int j = 0; j < rows * column; j++) {
                data[k][j] = (double) byteData[k][j];
            }
        }
    }
}
