package org.deeplearning4j.classifier.animals;

import org.jol.objects.MLItem;
import org.jol.objects.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Flower extends MLItem {

  public Flower(INDArray slice, MLModel model, String[] strings) {
    super(slice, model);
  }

}
