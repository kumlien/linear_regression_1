package se.kumliens.dl4j.logreg1;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import javax.annotation.PostConstruct;

import lombok.Data;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.LogLoss;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.impl.SigmoidUniformInitScheme;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.valueOf;
import static java.math.MathContext.DECIMAL64;
import static org.nd4j.linalg.api.buffer.DataType.DOUBLE;
import static org.nd4j.linalg.factory.Nd4j.ones;
import static org.nd4j.linalg.factory.Nd4j.sum;
import static org.nd4j.linalg.ops.transforms.Transforms.log;

/**
 * Common steps for pre-processing a new dataset are:
 *
 * Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
 * Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
 * "Standardize" the data
 */
@Component
@RequiredArgsConstructor
@Slf4j
@Data
public class Week2 {

    @Value("${classpath:/week2/train_catvnoncat.h5}")
    private Resource trainingSetResource;

    @Value("${classpath:/week2/test_catvnoncat.h5}")
    private Resource testSetResource;

    // 4d int array with test set. 209 images, three layers with R, G and B values
    // respectively
    int[][][][] trainSetXOrig;

    // Labels for the training set (Y)
    long[] trainSetYOrig;

    int[][][][] testSetXOrig;

    long[] testSetYOrig;

    //cat, non-cat
    String[] classes;

    //a.k.a X
    INDArray trainingSet;

    INDArray testSet;

    private INDArray w;

    private double b;


    /**
     * Problem Statement: You are given a dataset ("data.h5") containing:
     * 
     * - a training set of m_train images labeled as cat (y=1) or non-cat (y=0) - a
     * test set of m_test images labeled as cat or non-cat - each image is of shape
     * (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is
     * square (height = num_px) and (width = num_px).
     * 
     * You will build a simple image-recognition algorithm that can correctly
     * classify pictures as cat or non-cat.
     * 
     * @see <a href="https://deeplearning4j.konduit.ai/nd4j/overview">nd4j</>
     * @see <a href="https://github.com/jamesmudd/jhdf">jHDF - used to read
     *      hdf5-files</a>
     *
     * Load training and test sets
     */
    @PostConstruct
    public void init() {
        Map<String, Object> training_set = H5Reader.readH5(trainingSetResource);
        Map<String, Object> test_set = H5Reader.readH5(testSetResource);
        training_set.keySet().forEach(k -> Week2.log.info("TrainingSet with key {} and data {}", k,
                training_set.get(k).getClass().getSimpleName()));
        test_set.keySet().forEach(
                k -> Week2.log.info("TestSet with key {} and data {}", k, test_set.get(k).getClass().getSimpleName()));
        classes = (String[]) training_set.get("list_classes");
        //trainSetXOrig = new int[1][][][];
        //trainSetXOrig[0] = ((int[][][][]) training_set.get("train_set_x"))[0];
        trainSetXOrig = ((int[][][][]) training_set.get("train_set_x"));
        trainSetYOrig = (long[]) training_set.get("train_set_y");
        testSetXOrig = (int[][][][]) test_set.get("test_set_x");
        testSetYOrig = (long[]) test_set.get("test_set_y");
        reshapeTestSetAndStandardize();
        reshapeTrainingSetAndStandardize();
        initializeWeights();
    }

    public long[] reshapeTrainingSetAndStandardize() {
        long[] shape = new long[] {64*64*3, trainSetXOrig.length};
        trainingSet = reshapeImgDataAndStandardize(trainSetXOrig, shape);
        Week2.log.info("Trainingset has {} columns (images) and {} rows", trainingSet.columns(), trainingSet.rows());
        return trainingSet.shape();
    }

    public long[] reshapeTestSetAndStandardize() {
        long[] shape = new long[] {64*64*3, testSetXOrig.length};
        testSet = reshapeImgDataAndStandardize(testSetXOrig, shape);
        log.info("Testset has {} columns (images) and {} rows", testSet.columns(), testSet.rows());
        return testSet.shape();
    }


    /**
     * Reshape the provided data and 'standardize' it by dividing all values with 255.
     * Each column represents a flattened image.
     * It's the responsibility of the client to close the returned {@link INDArray}
     *
     * @param imgData four dim int array with image data (dim1=image, dim2, height, dim3 width, dim4 rgb-values)
     * @param newShape the new shape
     * @return A new (still open) {@link INDArray} with the reshaped data.
     */
    public INDArray reshapeImgDataAndStandardize(int[][][][] imgData, long[] newShape) {
        int[] flat = ArrayUtil.flatten(imgData);
        try (INDArray array = Nd4j.create(newShape, 'f')) {
            array.data().setData(flat);
            return array.div(255);
        }
    }

    /**
     * Do a forward propagation by calculating:
     * <ul>
     *     <li>weights * X + b</li>
     *     <li>sigmoid of the above</li>
     *     <li>loss of the above</li>
     *     <li>cost as the average of the losses</li>
     * </ul>
     */
    public void propagate() {
        INDArray Y = Nd4j.create(DOUBLE, trainSetYOrig.length,1);
        Y.data().setData(trainSetYOrig);
        doPropagate(w, b, trainingSet, Y);
    }

    public BigDecimal doPropagate(INDArray w, double b, INDArray X, INDArray Y) {
        BigDecimal oneOverM = valueOf(1).divide(valueOf(X.columns()), DECIMAL64);
        BigDecimal negOneOverM = oneOverM.negate();
        INDArray rawYHat = w.transpose().mmul(X).add(b);
        log.info("");
        log.info("");
        log.info("ActivationInput shape: {}", rawYHat.shape());
        INDArray A = sigmoid(rawYHat); //All activations, capital A
        log.info("A shape: {}", A.shape());
        log.info("A: {}", A.data());

        //start loss function
        //cost = -1/m*np.sum(Y*np.log(A)+ (1-Y)*np.log(1-A))
        //Left: Y*np.log(A)
        log.info("Y: {}", Y.data());
        INDArray logA = log(A);
        log.info("log(A): {}", logA.data());
        INDArray left = Y.transpose().mul(logA);
        log.info("Left: {}", left);

        //Right: (1-Y)*np.log(1-A)
        INDArray right = ones(1).sub(Y.transpose()).mul(log(ones(1).sub(A)));
        log.info("Right: {}", right);

        double sumLoss = left.add(right).sumNumber().doubleValue();
        log.info("Total Loss: {}", sumLoss);

        BigDecimal cost = negOneOverM.multiply(valueOf(sumLoss));
        log.info("Cost: {}", cost);


        //dw = (1 / m) * np.dot(X, (A - Y).T)
        //db = (1 / m) * np.sum(A - Y)
        log.info("Y.shape: {}", Y.shape());
        log.info("A.shape: {}", A.shape());
        INDArray diff = A.sub(Y.transpose());
        log.info("diff.shape: {}", A.shape());
        log.info("X.shape: {}", X.shape());

        INDArray dw = X.mmul(diff.transpose()).mul(oneOverM.doubleValue());
        log.info("dw: {}", dw.data());
        double db = oneOverM.multiply(valueOf(A.sub(Y.transpose()).sumNumber().doubleValue())).doubleValue();
        log.info("db: {}", db);

        return cost;
    }

    /**
     * f(x) = 1 / (1 + exp(-x))
     */
    public INDArray sigmoid(INDArray z) {
        return Transforms.sigmoid(z);
    }

    /**
     * Init weights with zeros with shape matching training set. Since it contains images it should be 3 * width * height
     */
    public void initializeWeights() {
        BaseWeightInitScheme initScheme = new SigmoidUniformInitScheme('c', trainSetXOrig[0].length * trainSetXOrig[0][0].length * trainSetXOrig[0][0][0].length, 1);
        w = initScheme.create(DOUBLE, trainSetXOrig[0].length * trainSetXOrig[0][0].length * trainSetXOrig[0][0][0].length, 1);
        b = 0d;
    }

    public int get_m_train() {
        return trainSetXOrig.length;
    }

    public int get_m_test() {
        return testSetXOrig.length;
    }

    public int getImageWidthAndHeigth() {
        return trainSetXOrig[0][0].length;
    }

    /**
     * Get the rgb-values for the training image with the specified index
     * 
     * @param ix
     */
    public int[] getTrainingSetImage(int ix) {
        if (ix > -1 && ix < trainSetXOrig.length) {
            Week2.log.info("Returning training image with ix {}", ix);
            return ArrayUtil.flatten(trainSetXOrig[ix]);
        }
        Week2.log.warn("Invalid index {}, must be a number between 0 and {}", ix, trainSetXOrig.length);
        return new int[] {};
    }
}