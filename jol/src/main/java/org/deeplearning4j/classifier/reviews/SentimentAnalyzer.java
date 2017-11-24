package org.deeplearning4j.classifier.reviews;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.utilities.DataUtilities;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jol.objects.MLConf;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;

public class SentimentAnalyzer {

//  public void fetchReviews (String input) throws Exception {
//    //Load the model
//    MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
//
//    //    System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
//    //    System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
//    INDArray features = test.loadFeaturesFromString(input, truncateReviewsToLength);
//    INDArray networkOutput = net.output(features);
//    int timeSeriesLength = networkOutput.size(2);
//    INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));
//
//    System.out.println("\n\n-------------------------------");
//    System.out.println("First positive review: \n" + input);
//    System.out.println("\n\nProbabilities at last time step:");
//    System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
//    System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));
//
//    System.out.println("----- Example complete -----");
//  }

  public static MultiLayerNetwork createModel(MLConf global_conf) throws Exception {
    if (global_conf.wordVectorsPath.startsWith("/PATH/TO/YOUR/VECTORS/")){
      throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
    }
    
    //Download and extract data
    downloadData(global_conf);

    Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

    
    //DataSetIterators for training and testing respectively
    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(global_conf.wordVectorsPath));
    SentimentExampleIterator train = new SentimentExampleIterator(global_conf.dataPath, wordVectors, global_conf.batchSize, global_conf.truncateReviewsToLength, true);
    SentimentExampleIterator test = new SentimentExampleIterator(global_conf.dataPath, wordVectors, global_conf.batchSize, global_conf.truncateReviewsToLength, false);

    //Set up network configuration
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .updater(Updater.ADAM)  //To configure: .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
    .regularization(true).l2(1e-5)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
    .learningRate(2e-2)
    .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(global_conf.vectorSize).nOut(256)
        .activation(Activation.TANH).build())
        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
            .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .pretrain(false).backprop(true).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(1));

    System.out.println("Starting training");
    for (int i = 0; i < global_conf.nEpochs; i++) {
      net.fit(train);
      train.reset();
      System.out.println("Epoch " + i + " complete. Starting evaluation:");
    }

    //Run evaluation. This is on 25k reviews, so can take some time
    Evaluation evaluation = net.evaluate(test);
    System.out.println(evaluation.stats());
    
    String firstPositiveReview = "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge.";

    INDArray features = test.loadFeaturesFromString(firstPositiveReview, global_conf.truncateReviewsToLength);
    INDArray networkOutput = net.output(features);
    int timeSeriesLength = networkOutput.size(2);
    INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

    System.out.println("\n\n-------------------------------");
    System.out.println("First positive review: \n" + firstPositiveReview);
    System.out.println("\n\nProbabilities at last time step:");
    System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
    System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

    System.out.println("----- Example complete -----");
    
    return net;
  }

  public static void downloadData(MLConf global_conf) throws Exception {
    //Create directory if required
    File directory = new File(global_conf.dataPath);
    if(!directory.exists()) directory.mkdir();

    //Download file:
    String archizePath = global_conf.dataPath + "aclImdb_v1.tar.gz";
    File archiveFile = new File(archizePath);
    String extractedPath = global_conf.dataPath + "aclImdb";
    File extractedFile = new File(extractedPath);

    if( !archiveFile.exists() ){
      System.out.println("Starting data download (80MB)...");
      FileUtils.copyURLToFile(new URL(global_conf.dataUrl), archiveFile);
      System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
      //Extract tar.gz file to output directory
      DataUtilities.extractTarGz(archizePath, global_conf.dataPath);
    } else {
      //Assume if archive (.tar.gz) exists, then data has already been extracted
      System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
      if( !extractedFile.exists()){
        //Extract tar.gz file to output directory
        DataUtilities.extractTarGz(archizePath, global_conf.dataPath);
      } else {
        System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
      }
    }
  }
}
