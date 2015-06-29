package org.grobid.trainer;

import com.google.common.base.Joiner;
import org.allenai.ml.sequences.crf.conll.ConllFormat;
import org.allenai.ml.sequences.crf.conll.Trainer;
import org.grobid.core.GrobidModels;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import static org.allenai.ml.util.IOUtils.linesFromPath;

public class AI2CRFGenericTrainer implements GenericTrainer {
    public static final Logger LOGGER = LoggerFactory.getLogger(AI2CRFGenericTrainer.class);
    public static final String CRF = "ai2-ml";
    private final org.allenai.ml.sequences.crf.conll.Trainer trainer;

    // default training parameters (not exploited by CRFPP so far, it requires to extend the JNI)
    private double epsilon = 0.00001; // default size of the interval for stopping criterion
    private int window = 20; // default similar to CRF++

    public AI2CRFGenericTrainer() {
        trainer = new Trainer();
    }

    @Override
    public void train(File template, File trainingData, File outputModel, int numThreads, GrobidModels model) {

        List<List<ConllFormat.Row>> labeledData =
            ConllFormat.readData(linesFromPath(trainingData.getAbsolutePath()), true);
        List<List<ConllFormat.Row>> nonBrokenLabeledData = new ArrayList<List<ConllFormat.Row>>();
        for (List<ConllFormat.Row> rows : labeledData) {
            if (rows.get(1).getLabel().get().startsWith("I-")) {
                nonBrokenLabeledData.add(rows);
            }
        }
        String fixedTrainDataPath = null;
        try {
            File fixedTrainData = File.createTempFile("fixed-train", "train");
            BufferedWriter writer = new BufferedWriter(new FileWriter(fixedTrainData.getAbsoluteFile()));
            for (List<ConllFormat.Row> rows : nonBrokenLabeledData) {
                for (ConllFormat.Row row : rows) {
                    String feats = Joiner.on("\t").join(row.features);
                    writer.write(feats);
                    writer.write("\t" + row.getLabel().get());
                    writer.write("\n");
                }
                writer.write("\n");
            }
            writer.close();
            fixedTrainDataPath = fixedTrainData.getAbsolutePath();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        System.out.printf("Original data size: %d, fixed data size: %d");
        Trainer.Opts opts = new Trainer.Opts();
        opts.templateFile = template.getAbsolutePath();
        opts.trainPath = fixedTrainDataPath;
        opts.modelPath = outputModel.getAbsolutePath();
        opts.numThreads = numThreads;
        opts.featureKeepProb = 0.1;
        opts.maxIterations = 300;

        trainer.trainAndSaveModel(opts);
    }

    @Override
    public String getName() {
        return CRF;
    }

    @Override
    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public void setWindow(int window) {
        this.window = window;
    }

    @Override
    public double getEpsilon() {
        return epsilon;
    }

    @Override
    public int getWindow() {
        return window;
    }
}
