package se.kumliens.dl4j.logreg1;

import java.util.Map;

import javax.annotation.PostConstruct;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import java.awt.event.ActionEvent;

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

    private final H5Reader h5Reader;

    @Value("${classpath:/week2/train_catvnoncat.h5}")
    private Resource trainingSet;

    @Value("${classpath:/week2/test_catvnoncat.h5}")
    private Resource testSet;


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
     */
    public int[] getImageData(int imagenumber) {
        Map<String, Object> training_set = h5Reader.readH5(trainingSet);
        Map<String, Object> test_set = h5Reader.readH5(testSet);

        training_set.keySet().forEach(k -> log.info("TrainingSet with key {} and data {}", k,
                training_set.get(k).getClass().getSimpleName()));
        test_set.keySet().forEach(
                k -> log.info("TestSet with key {} and data {}", k, test_set.get(k).getClass().getSimpleName()));

        String[] classes = (String[]) training_set.get("list_classes");
        int[][][][] trainSetXOrig = (int[][][][]) training_set.get("train_set_x");
        long[] trainSetYOrig = (long[]) training_set.get("train_set_y");

        int[][][][] testSetXOrig = (int[][][][]) test_set.get("test_set_x");
        long[] testSetYOrig = (long[]) test_set.get("test_set_y");

        int[] flat = ArrayUtil.flatten(trainSetXOrig);
        log.info("Number of training images: {}", trainSetXOrig.length);
        int[] shape = new int[] { trainSetXOrig.length, 3, 64, 64, 64 };
        INDArray myArr = Nd4j.create(flat, shape, 'c');
        log.info("The shape of the INDArray is {}", myArr.shapeDescriptor());
        return ArrayUtil.flatten(trainSetXOrig[imagenumber]);
    }
}