package se.kumliens.dl4j.logreg1;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import io.jhdf.api.Node;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

/**
 * Used lib from https://github.com/jamesmudd/jhdf for parsing h5
 */
@Slf4j
public class H5Reader {

    @SneakyThrows
    public static Map<String, Object> readH5(Resource resource) {
        log.info("Loading h5 data from {}", resource.getURL());
 
        Map<String, Object> datasets = new HashMap<>();
        try (HdfFile hdfFile = HdfFile.fromInputStream(resource.getInputStream())) {
            Map<String, Node> children = hdfFile.getChildren();
            children.keySet().forEach(k -> {
                Dataset ds = hdfFile.getDatasetByPath(children.get(k).getPath());
                datasets.put(k, ds.getData());
                log.info("Adding dataset {} with path {} and dims {} of type {}", k, children.get(k).getPath(),
                        ds.getDimensions(), ds.getJavaType().getName());
            });
        }
        return datasets;
    }

}