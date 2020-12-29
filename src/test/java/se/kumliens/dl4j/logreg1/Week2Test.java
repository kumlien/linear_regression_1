package se.kumliens.dl4j.logreg1;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(SpringExtension.class)
@ContextConfiguration(classes = TestConfig.class)
class Week2Test {

    @Autowired
    Week2 week2;

    int noOfTrainImages = 209;

    int noOfTestImages = 50;

    int imgSize = 3 * 64 * 64;

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
}