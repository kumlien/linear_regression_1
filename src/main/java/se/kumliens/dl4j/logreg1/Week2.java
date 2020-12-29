package se.kumliens.dl4j.logreg1;

import java.util.Map;

import javax.annotation.PostConstruct;

import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Component
@RequiredArgsConstructor
@Slf4j
public class Week2 {

    @Value("${classpath:/week2/train_catvnoncat.h5}")
    private Resource trainingSetResource;

    @Value("${classpath:/week2/test_catvnoncat.h5}")
    private Resource testSetResource;

    // 4d int array with test set. 209 images, three layers with R, G and B values
    // respectively
    int[][][][] trainSetXOrig;

    // Labels for the training set
    long[] trainSetYOrig;

    int[][][][] testSetXOrig;

    long[] testSetYOrig;

    //cat, non-cat
    String[] classes;

    INDArray trainingSet;

    INDArray testSet;


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
     *      Load training and test sets
     */
    @PostConstruct
    public void init() {
        Map<String, Object> training_set = H5Reader.readH5(trainingSetResource);
        Map<String, Object> test_set = H5Reader.readH5(testSetResource);
        training_set.keySet().forEach(k -> log.info("TrainingSet with key {} and data {}", k,
                training_set.get(k).getClass().getSimpleName()));
        test_set.keySet().forEach(
                k -> log.info("TestSet with key {} and data {}", k, test_set.get(k).getClass().getSimpleName()));
        classes = (String[]) training_set.get("list_classes");
        trainSetXOrig = (int[][][][]) training_set.get("train_set_x");
        trainSetYOrig = (long[]) training_set.get("train_set_y");
        testSetXOrig = (int[][][][]) test_set.get("test_set_x");
        testSetYOrig = (long[]) test_set.get("test_set_y");
    }

    public long[] reshapeTrainingSetAndStandardize() {
        int[] flat = ArrayUtil.flatten(trainSetXOrig);
        log.info("Number of training images: {}", trainSetXOrig.length);
        long[] shape = new long[] {64*64*3, trainSetXOrig.length};
        try (INDArray array = Nd4j.create(flat, shape, DataType.INT32)) {
            trainingSet = array.div(255);
        }
        log.info("The shape of the training set is {}", trainingSet.shapeDescriptor());
        return trainingSet.shape();
    }

    public long[] reshapeTestSetAndStandardize() {
        int[] flat = ArrayUtil.flatten(testSetXOrig);
        log.info("Number of test images: {}", testSetXOrig.length);
        long[] shape = new long[] {64*64*3, testSetXOrig.length};
        try (INDArray array = Nd4j.create(flat, shape, DataType.INT32)) {
            testSet = array.div(255);
        }
        log.info("The shape of the test set is {}", testSet.shapeDescriptor());
        return testSet.shape();
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
            return ArrayUtil.flatten(trainSetXOrig[ix]);
        }
        log.warn("Invalid index {}, must be a number between 0 and {}", ix, trainSetXOrig.length);
        return new int[] {};
    }
}