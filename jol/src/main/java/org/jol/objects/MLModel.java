package org.jol.objects;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.classifier.animals.BasicCSVClassifier;
import org.deeplearning4j.classifier.reviews.SentimentExampleIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.utilities.DataUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MLModel {

  private Model model;
  private MLConf conf;

  private WordVectors wvs;
  private DefaultTokenizerFactory tokenizerFactory;
  private DataNormalization normalizer = new NormalizerStandardize();

  public MLModel (MLConf mlConf) throws Exception {
    conf = mlConf;
   // wvs = WordVectorSerializer.loadStaticModel(new File(conf.wordVectorsPath));

    if (conf.create) {
     // model = SentimentAnalyzer.createModel(conf);
      model = BasicCSVClassifier.createModel(conf);
      saveToDisk();
    }
    else if (conf.type.equals("dl4j")) {
      model = ModelSerializer.restoreMultiLayerNetwork(conf.modelLocation);
    }
    
    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    
    DataSet testData = DataUtilities.readCSVDataset("/DataExamples/animals/animals.csv",
        conf.batchSizeTest, conf.labelIndex, conf.numClasses);
    normalizer.fit(testData);
  }

  public void saveToDisk() throws IOException  {
    ModelSerializer.writeModel(model, conf.modelLocation, true);
  }
  
  public void normalize (INDArray slice) {
    normalizer.transform(slice);
  }

  public INDArray prepareFeatures(String input) throws IOException {
    return prepareFeatures(input, conf.truncateReviewsToLength, wvs);
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

  public INDArray getOutput(INDArray features) {
    return ((MultiLayerNetwork)model).output(features);
  }

  public String eval() throws IOException {
    SentimentExampleIterator test = new SentimentExampleIterator(conf.dataPath, wvs, conf.batchSize, conf.truncateReviewsToLength, false);
    Evaluation evaluation = ((MultiLayerNetwork)model).evaluate(test);
    return evaluation.stats();
  }
}
