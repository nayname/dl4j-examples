package org.jol;

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

}
