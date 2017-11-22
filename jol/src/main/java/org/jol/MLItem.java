package org.jol;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MLItem {
  
  MLModel model;
  String label;

  public MLItem(String input, MLModel model_) throws Exception{
    model = model_;
    //    System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
    //    System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
    INDArray networkOutput = model.feed(input);

    int timeSeriesLength = networkOutput.size(2);
    INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

    if (probabilitiesAtLastWord.getDouble(0) > probabilitiesAtLastWord.getDouble(1))
      label = model.getLabels().get(1);
    else
      label = model.getLabels().get(0);
  }

  public String getLabel() {
    return label;
  }
}
