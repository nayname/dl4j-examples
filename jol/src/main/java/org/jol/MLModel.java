package org.jol;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.sentiment.reviews.SentimentAnalyzer;
import org.deeplearning4j.sentiment.reviews.SentimentExampleIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MLModel {

  private Model model;
  private MLConf conf;

  private WordVectors wvs;
  private DefaultTokenizerFactory tokenizerFactory;

  public MLModel (MLConf mlConf) throws Exception {
    conf = mlConf;
    wvs = WordVectorSerializer.loadStaticModel(new File(conf.wordVectorsPath));
    
    if (conf.create) {
      model = SentimentAnalyzer.createModel(conf);
      saveToDisk();
    }
    else if (conf.type.equals("dl4j")) {
      model = ModelSerializer.restoreMultiLayerNetwork(conf.modelLocation);
    }

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
  }

  public void saveToDisk() throws IOException  {
    ModelSerializer.writeModel(model, conf.modelLocation, true);
  }

  public INDArray feed(String input) throws IOException {
    INDArray features = prepareFeatures(input, conf.truncateReviewsToLength, wvs);
    return((MultiLayerNetwork)model).output(features);
  }

  /**
   * TODO: overriden from org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
   * @param reviewContents
   * @param maxLength
   * @param wordVectors
   * @return
   */
      
  public INDArray prepareFeatures(String reviewContents, int maxLength, WordVectors wordVectors){
    int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    List<String> tokens = tokenizerFactory.create(reviewContents).getTokens();
    List<String> tokensFiltered = new ArrayList<>();
    
    for(String t : tokens ){
      if(wordVectors.hasWord(t)) tokensFiltered.add(t);
    }
    int outputLength = Math.max(maxLength,tokensFiltered.size());

    INDArray features = Nd4j.create(1, vectorSize, outputLength);
    for( int j=0; j<tokens.size() && j<maxLength; j++ ){
      String token = tokens.get(j);
      INDArray vector = wordVectors.getWordVectorMatrix(token);
      features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
    }

    return features;
  }
  
  public List<String> getLabels() {
    return Arrays.asList("positive","negative");
  }
}
