package se.kumliens.dl4j.logreg1;

import java.util.Map;

import javax.annotation.PostConstruct;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Component;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Component
@RequiredArgsConstructor
@Slf4j
public class Week2 {

    private final H5Reader h5Reader;

    /**
     * Problem Statement: You are given a dataset ("data.h5") containing:

     * - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
     * - a test set of m_test images labeled as cat or non-cat
     * - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
     * 
     * You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.
     * 
     * @see <a href="https://deeplearning4j.konduit.ai/nd4j/overview">nd4j</>
     */
    @PostConstruct
    public void init() {
        Map<String, Object> training_set = h5Reader.readH5("classpath:week2/train_catvnoncat.h5");
        Map<String, Object> test_set = h5Reader.readH5("classpath:week2/test_catvnoncat.h5");

        training_set.keySet().forEach(k -> log.info("TrainingSet with key {} and data {}", k, training_set.get(k).getClass().getSimpleName()));
        test_set.keySet().forEach(k -> log.info("TestSet with key {} and data {}", k, test_set.get(k).getClass().getSimpleName()));

        String[] classes = (String[]) training_set.get("list_classes");
        int[][][][] trainSetXOrig = (int[][][][]) training_set.get("train_set_x");
        long[] trainSetYOrig = (long[]) training_set.get("train_set_y");

        int[][][][] testSetXOrig = (int[][][][]) test_set.get("test_set_x");
        long[] testSetYOrig = (long[]) test_set.get("test_set_y");

        INDArray i;
        Nd4j.zeros(1,1,5);
        INDArray nTrainSetOrig = Nd4j.create(trainSetXOrig);
        log.info("nTrainSetOrig: {}", nTrainSetOrig.data().asInt());
    
    }
}