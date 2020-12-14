package se.kumliens.dl4j.logreg1;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

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
@Service
@Slf4j
@RequiredArgsConstructor
public class H5Reader {

    private final ResourceLoader resourceLoader;

    @SneakyThrows
    public Map<String, Object> readH5(String fileResource) {
        log.info("Loading h5 data from {}", fileResource);
        File file = resourceLoader.getResource(fileResource).getFile();
        log.info("File {} exists: {}", file.getAbsolutePath(), file.exists());
        Map<String, Object> datasets = new HashMap<>();
        try (HdfFile hdfFile = new HdfFile(file)) {
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