package org.deeplearning4j.classifier.animals;


import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.utilities.DataUtilities;
import org.jol.objects.MLConf;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * This example is intended to be a simple CSV classifier that seperates the training data
 * from the test data for the classification of animals. It would be suitable as a beginner's
 * example because not only does it load CSV data into the network, it also shows how to extract the
 * data and display the results of the classification, as well as a simple method to map the lables
 * from the testing data into the results.
 *
 * @author Clay Graham
 */
public class BasicCSVClassifier {

  private static Logger log = LoggerFactory.getLogger(BasicCSVClassifier.class);

  public static MultiLayerNetwork createModel(MLConf global_conf) throws Exception {
    DataSet trainingData = DataUtilities.readCSVDataset(
        "/DataExamples/animals/animals_train.csv",
        global_conf.batchSizeTraining, global_conf.labelIndex, global_conf.numClasses);


    DataSet testData = DataUtilities.readCSVDataset("/DataExamples/animals/animals.csv",
        global_conf.batchSizeTest, global_conf.labelIndex, global_conf.numClasses);

    //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
    DataNormalization normalizer = new NormalizerStandardize();
    normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
    normalizer.transform(trainingData);     //Apply normalization to the training data
    normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

    log.info("Build model....");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(global_conf.seed)
    .iterations(global_conf.iterations)
    .activation(Activation.TANH)
    .weightInit(WeightInit.XAVIER)
    .learningRate(0.1)
    .regularization(true).l2(1e-4)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(global_conf.numInputs).nOut(3).build())
    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
    .activation(Activation.SOFTMAX).nIn(3).nOut(global_conf.outputNum).build())
    .backprop(true).pretrain(false)
    .build();

    //run the model
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(new ScoreIterationListener(100));

    System.out.println("Starting training");
    for (int i = 0; i < global_conf.nEpochs; i++) {
      model.fit(trainingData);
      //      trainingData.reset();
      System.out.println("Epoch " + i + " complete. Starting evaluation:");
    }

    INDArray output = model.output(testData.getFeatureMatrix());

    //evaluate the model on the test set
    Evaluation eval = new Evaluation(3);
    eval.eval(testData.getLabels(), output);

    System.out.println(eval.stats());

    return model;
  }
}
