package se.kumliens.dl4j.logreg1;

import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ApplicationEvent;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.support.GenericApplicationContext;

import javafx.application.Application;
import javafx.application.HostServices;
import javafx.application.Platform;
import javafx.stage.Stage;

public class JavaFXApplication extends Application {

    private ConfigurableApplicationContext ctx;

    @Override
    public void init() throws Exception {
        // Create initializer which will add some nice-to-have beans to the context
        ApplicationContextInitializer<GenericApplicationContext> initializer = ac -> {
            ac.registerBean(Application.class, () -> JavaFXApplication.this);
            ac.registerBean(Parameters.class, this::getParameters);
            ac.registerBean(HostServices.class, this::getHostServices);
        };

        // Init Sping
        ctx = new SpringApplicationBuilder().sources(NeuralNetworksAndDeepLearning.class).initializers(initializer)
                .run(getParameters().getRaw().toArray(new String[0]));
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        ctx.publishEvent(new JFXStageReadyEvent(primaryStage));

    }

    @Override
    public void stop() throws Exception {
        ctx.close();
        Platform.exit();
    }

    public static class JFXStageReadyEvent extends ApplicationEvent {
    
        public JFXStageReadyEvent(Stage source) {
            super(source);
        }

        public Stage getStage() {
            return Stage.class.cast(source);
        }

    }

}