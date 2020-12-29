package se.kumliens.dl4j.logreg1.ui;

import java.io.IOException;
import java.net.URL;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationListener;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import lombok.RequiredArgsConstructor;
import se.kumliens.dl4j.logreg1.JavaFXApplication.JFXStageReadyEvent;

@Component
@RequiredArgsConstructor
public class UI implements ApplicationListener<JFXStageReadyEvent> {

    @Value("${app.ui.title}")
    private String title;

    @Value("classpath:/ui.fxml")
    private Resource fxml;

    private final ApplicationContext appCtx;

    @Override
    public void onApplicationEvent(JFXStageReadyEvent event) {
        Stage stage = event.getStage();
        try {
            URL url = fxml.getURL();
            FXMLLoader loader = new FXMLLoader(url);
            loader.setControllerFactory(appCtx::getBean);
            Parent root = loader.load();
            Scene scene = new Scene(root, 600, 600);
            stage.setScene(scene);
            stage.setTitle(title);
            stage.show();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}