package org.jol;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MLItem {
  
  MLModel model;

  public MLItem(String input, MLModel model_, String type){
    model = model_;
    //    System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
    //    System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
    INDArray networkOutput = model.feed(input);
    int timeSeriesLength = networkOutput.size(2);
    INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

    System.out.println("\n\n-------------------------------");
    System.out.println("First positive review: \n" + input);
    System.out.println("\n\nProbabilities at last time step:");
    System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
    System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

    System.out.println("----- Example complete -----");
  }
  
  public double[] getFeatures() {
    return features;
  }
  
  public void setScore(float f) {
    score = f;
  }
  
  public Float getScore() {
    return score;
  }
  
  private static double getFeatureVal(int i, ItemData item, SearchConfiguration conf) {
    double val = Double.parseDouble(String.valueOf(item.features.get(conf.getMlFeatsOrder().get(i))));
    MLFeat ml = conf.getMlFeats().get(conf.getMlFeatsOrder().get(i));
    return ((val-ml.scaling.get(0))/ml.scaling.get(1));
  }

  public Object getSource() {
    return source;
  }

  public String getLabel() {
    // TODO Auto-generated method stub
    return null;
  }
}
