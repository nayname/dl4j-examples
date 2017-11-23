package org.jol;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MLItem {
  
  MLModel model;
  
  private INDArray features;
  private INDArray output;

  public MLItem(String input, MLModel model_) throws Exception{
    model = model_;
    features = model.prepareFeatures(input);
    
    output = model.getOutput(features);
  }

//  public String getLabel() {
//    return label;
//  }
  
  public INDArray getOutput() {
    return output;
  }
  
  public String eval() throws IOException {
    return model.eval();
  }
}
