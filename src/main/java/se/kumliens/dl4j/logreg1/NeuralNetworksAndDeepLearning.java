package se.kumliens.dl4j.logreg1;

import org.springframework.boot.autoconfigure.SpringBootApplication;

import javafx.application.Application;

@SpringBootApplication
public class NeuralNetworksAndDeepLearning {

	public static void main(String[] args) {
		//Launch the JavaFX app which will wire in the Spring context
		Application.launch(JavaFXApplication.class, args);
	}
}