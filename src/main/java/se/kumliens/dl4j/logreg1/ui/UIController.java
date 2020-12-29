package se.kumliens.dl4j.logreg1.ui;

import javafx.application.HostServices;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import se.kumliens.dl4j.logreg1.Week2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


@Component
@Slf4j
@RequiredArgsConstructor
public class UIController {

    private final HostServices hostServices;

    private final Week2 week2;

    @FXML
    public Button button;

    @FXML
    public TextField imagenumber;

    @FXML
    public ImageView imageView;

    @FXML
    public void initialize() {
        button.setOnAction(event -> {
            Integer text = Integer.valueOf(imagenumber.getText());
            int[] pixels = week2.getImageData(text);
            byte[] pixelBytes = new byte[pixels.length];
            for(int i=0; i<pixels.length; i++) {
                pixelBytes[i] = (byte) pixels[i];
            }
            int width = 64;
            int height = 64;
            WritableImage img = new WritableImage(width, height);
            PixelWriter pw = img.getPixelWriter();
            pw.setPixels(0, 0, width, height, PixelFormat.getByteRgbInstance(), pixelBytes, 0, width*3);
            imageView.setImage(img);
        });
    }
}


