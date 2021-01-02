package se.kumliens.dl4j.logreg1;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(SpringExtension.class)
@ContextConfiguration(classes = TestConfig.class)
class Week2Test {

    @Autowired
    Week2 week2;

    int noOfTrainImages = 209;

    int noOfTestImages = 50;

    int imgSize = 3 * 64 * 64;

    double epsilon = 0.0001d;

    @Test
    @DisplayName("Verify correct reshaping of training data")
    public void testReshapeTrainingSet() {
        long[] shape = week2.reshapeTrainingSetAndStandardize();
        assertEquals(2, shape.length, "The shape should have two dims");
        assertEquals(imgSize, shape[0]);
        assertEquals(noOfTrainImages, shape[1]);
    }

    @Test
    @DisplayName("Verify correct reshaping of test data")
    public void testReshapeTestSet() {
        long[] shape = week2.reshapeTestSetAndStandardize();
        assertEquals(2, shape.length, "The shape should have two dims");
        assertEquals(imgSize, shape[0]);
        assertEquals(noOfTestImages, shape[1]);
    }

    @Test
    @DisplayName("Verify correct reshaping of img data for three 3*3 rgb images")
    public void testReshape(){
        int[][][][] imgData = new int[][][][]{
                {new int[][]{{0,1,2},{3,4,5}, {6,7,8}}},
                {new int[][]{{10,11,12},{13,14,15}, {16,17,18}}},
                {new int[][]{{20,21,22},{23,24,25}, {26,27,28}}},
        };
        long[] newShape = new long[]{3*3, imgData.length};
        try (INDArray result = week2.reshapeImgDataAndStandardize(imgData, newShape)) {
            for(int imgIx=0; imgIx<imgData.length; imgIx++) {
                INDArray c1 = result.getColumn(imgIx);
                System.out.println("Check img " + imgIx + " with data " + c1);
                for(int i=0; i<9; i++) {
                    double d = c1.getDouble(i);
                    assertEquals(Double.valueOf(i + (imgIx*10)) / 255, d, epsilon);
                }
            }
        }
    }

    @Test
    @DisplayName("Verify number of images in training set")
    void get_m_train() {
        assertEquals(209, week2.get_m_train());
    }

    @Test
    @DisplayName("Verify number of images in test set")
    void get_m_test() {
        assertEquals(50, week2.get_m_test());
    }

    @Test
    @DisplayName("Verify height (and width) of the images")
    void get_num_px() {
        assertEquals(64, week2.getImageWidthAndHeigth());
    }

    @Test
    @DisplayName("Verify correct sigmoid behaviour")
    public void testSigmoid() {
        INDArray z = Nd4j.createFromArray(new double[]{0,2});
        INDArray s = week2.sigmoid(z);
        assertEquals(0.5, s.data().getDouble(0), epsilon);
        assertEquals(0.8808, s.data().getDouble(1), epsilon);
    }

    @Test
    @DisplayName("Make sure weights are initialized")
    public void testWeightInit() {
        week2.initializeWeights();
        assertEquals(0, week2.getB());
        assertEquals(3*64*64, week2.getW().shape()[0]);
        assertEquals(1, week2.getW().shape()[1]);
    }

    @Test
    @DisplayName("Verify the forward prop step")
    public void testForwardProp() {
        INDArray w = Nd4j.create(new double[]{1.0, 2.0}, new int[]{2, 1});
        double b = 2.0;
        INDArray X = Nd4j.create(new double[][]{{1.0, 2.0, -1.0},{3.0, 4.0, -3.2}});
        INDArray Y = Nd4j.create(DataType.DOUBLE, 3,1);
        Y.data().setData(new int[]{1,0,1});
        BigDecimal cost = week2.doPropagate(w, b, X, Y);
        assertEquals(5.80181093592113, cost.doubleValue(), epsilon);
        week2.propagate();
    }
}