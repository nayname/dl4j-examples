package org.jol.objects;

import java.util.Map;

import org.deeplearning4j.utilities.DataUtilities;
import org.nd4j.linalg.dataset.DataSet;

public class MLConf {

  /***
   * TODO: JSON input
   */

  public String dataPath;
  public String modelLocation;
  public String wordVectorsPath;
  public String dataUrl;

  public int batchSize;
  public int vectorSize;
  public int nEpochs;
  public int truncateReviewsToLength;
  public String type;

  boolean save = false;
  public boolean create = false;
  public int seed = 6;
  public int iterations = 1000;
  public int numInputs = 4;
  public int outputNum = 3;

  //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
  public int labelIndex = 4;     //5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
  public int numClasses = 3;     //3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2

  public int batchSizeTraining = 30;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
  //this is the data we want to classify
  public int batchSizeTest = 44;
}



