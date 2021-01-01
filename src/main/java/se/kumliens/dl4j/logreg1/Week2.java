package se.kumliens.dl4j.logreg1;

import java.util.Map;

import javax.annotation.PostConstruct;

import org.nd4j.common.util.ArrayUtil;
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
     * Load training and test sets
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
        reshapeTestSetAndStandardize();
        reshapeTrainingSetAndStandardize();
    }

    public long[] reshapeTrainingSetAndStandardize() {
        long[] shape = new long[] {64*64*3, trainSetXOrig.length};
        trainingSet = reshapeImgDataAndStandardize(trainSetXOrig, shape);
        log.info("Trainingset has {} columns (images) and {} rows", trainingSet.columns(), trainingSet.rows());
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
     * It's the responsibility of the client to close the returned {@link INDArray}
     *
     * @param imgData four dim int array with image data
     * @param newShape the new shape
     * @return A new (still open) {@link INDArray} with the reshaped data.
     */
    public INDArray reshapeImgDataAndStandardize(int[][][][] imgData, long[] newShape) {
        int[] flat = ArrayUtil.flatten(imgData);
        Nd4j.create(newShape, 'c');
        try (INDArray array = Nd4j.create(newShape, 'f')) {
            array.data().setData(flat);
            return array.div(255);
        }
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
            log.info("Returning training image with ix {}", ix);
            return ArrayUtil.flatten(trainSetXOrig[ix]);
        }
        log.warn("Invalid index {}, must be a number between 0 and {}", ix, trainSetXOrig.length);
        return new int[] {};
    }
}