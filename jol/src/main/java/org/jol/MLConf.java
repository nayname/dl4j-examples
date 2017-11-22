package org.jol;

public class MLConf {
  
  /***
   * TODO: JSON input
   */
  
  public String dataPath;
  String modelLocation;
  String wordVectorsPath;
  
  String model_builder;
  String type;
  
  int truncateReviewsToLength;
  
  boolean save = false;
  public boolean create = false;

  public MLConf(String dataPath_, String modelLocation_, String wordVectorsPath_,
      boolean save_) {
    dataPath = dataPath_;
    modelLocation = modelLocation_;
    wordVectorsPath = wordVectorsPath_;
    save = save_;
  }

}
