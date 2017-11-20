package org.deeplearning4j.sentiment.reviews;

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

  public String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
  public String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
  public String WORD_VECTORS_PATH = "/home/nayname/dl4j-examples/jol/sentiment/GoogleNews-vectors-negative300.bin.gz";
  public File locationToSave = new File("/home/nayname/dl4j-examples/jol/sentiment/sentiment_model.zip");
  public boolean create = false;

  public int batchSize = 64;     //Number of examples in each minibatch
  public int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
  public int nEpochs = 5;        //Number of epochs (full passes of training data) to train on
  public int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

  public WordVectors wordVectors;
  public SentimentExampleIterator test;

  private Logger logger;

  public SentimentAnalyzer() throws Exception {
    //DataSetIterators for training and testing respectively
    wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
    test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);
    logger = LoggerFactory.getLogger(SentimentAnalyzer.class);
  }

  public void fetchReviews (String input) throws Exception {
    //Load the model
    MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

    //    System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
    //    System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
    INDArray features = test.loadFeaturesFromString(input, truncateReviewsToLength);
    INDArray networkOutput = net.output(features);
    int timeSeriesLength = networkOutput.size(2);
    INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

    System.out.println("\n\n-------------------------------");
    System.out.println("First positive review: \n" + input);
    System.out.println("\n\nProbabilities at last time step:");
    System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
    System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

    System.out.println("----- Example complete -----");
  }

  public void createModel() throws Exception {
    if(WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")){
      throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
    }

    //Download and extract data
    downloadData();

    Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

    //Set up network configuration
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .updater(Updater.ADAM)  //To configure: .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
    .regularization(true).l2(1e-5)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
    .learningRate(2e-2)
    .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
        .activation(Activation.TANH).build())
        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
            .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .pretrain(false).backprop(true).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(1));

    //DataSetIterators for training and testing respectively
    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
    SentimentExampleIterator train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
    SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

    System.out.println("Starting training");
    for (int i = 0; i < nEpochs; i++) {
      net.fit(train);
      train.reset();
      System.out.println("Epoch " + i + " complete. Starting evaluation:");

      evaluate(test);
    }

    //Save the model
    boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, saveUpdater);
  }

  public void evaluate (SentimentExampleIterator data) throws Exception {
    MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
    Evaluation evaluation = net.evaluate(data);
    logger.info(evaluation.stats());
  }

  public void downloadData() throws Exception {
    //Create directory if required
    File directory = new File(DATA_PATH);
    if(!directory.exists()) directory.mkdir();

    //Download file:
    String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
    File archiveFile = new File(archizePath);
    String extractedPath = DATA_PATH + "aclImdb";
    File extractedFile = new File(extractedPath);

    if( !archiveFile.exists() ){
      System.out.println("Starting data download (80MB)...");
      FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
      System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
      //Extract tar.gz file to output directory
      DataUtilities.extractTarGz(archizePath, DATA_PATH);
    } else {
      //Assume if archive (.tar.gz) exists, then data has already been extracted
      System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
      if( !extractedFile.exists()){
        //Extract tar.gz file to output directory
        DataUtilities.extractTarGz(archizePath, DATA_PATH);
      } else {
        System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
      }
    }
  }
}
