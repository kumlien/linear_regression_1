package se.kumliens.dl4j.logreg1;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class TestConfig {

    @Bean
    public Week2 week2() {
        return new Week2();
    }
}
