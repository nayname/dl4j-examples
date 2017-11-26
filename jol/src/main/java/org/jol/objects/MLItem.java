package org.jol.objects;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.utilities.DataUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MLItem {
  
  MLModel model;
  
  private INDArray features;
  private INDArray output;
  
  private HashMap<String, Object> params = new HashMap<>();
  private String label;
  
  private static Map<Integer,String> eats = DataUtilities.readEnumCSV("/DataExamples/animals/eats.csv");
  private static Map<Integer,String> sounds = DataUtilities.readEnumCSV("/DataExamples/animals/sounds.csv");
  private static Map<Integer,String> classifiers = DataUtilities.readEnumCSV("/DataExamples/animals/classifiers.csv");

  public MLItem(String input, MLModel model_) throws Exception{
    this(model_.prepareFeatures(input), model_);
  }

  public MLItem(INDArray features_, MLModel model_) {
    model = model_;
    features = features_;
    
    setParams();
    model.normalize(features_);
    
    output = model.getOutput(features);
  }

  private void setParams() {
    addParam("yearsLived", features.getInt(0));
    addParam("eats", eats.get(features.getInt(1)));
    addParam("sounds", sounds.get(features.getInt(2)));
    addParam("weight", features.getFloat(3));
  }

  public String getLabel() {
    label = classifiers.get(maxIndex(getFloatArrayFromSlice(output)));
    return label;
  }
  
  public INDArray getOutput() {
    return output;
  }
  
  public String eval() throws IOException {
    return model.eval();
  }
  
  public void addParam (String key, Object value) {
    params.put(key, value);
  }
  
  public String toString() {
    return "params:"+params+", label: "+label;
  }
  
  /**
   * This method is to show how to convert the INDArray to a float array. This is to
   * provide some more examples on how to convert INDArray to types that are more java
   * centric.
   *
   * @param rowSlice
   * @return
   */
  public static float[] getFloatArrayFromSlice(INDArray rowSlice){
    float[] result = new float[rowSlice.columns()];
    for (int i = 0; i < rowSlice.columns(); i++) {
      result[i] = rowSlice.getFloat(i);
    }
    return result;
  }

  /**
   * find the maximum item index. This is used when the data is fitted and we
   * want to determine which class to assign the test row to
   *
   * @param vals
   * @return
   */
  public static int maxIndex(float[] vals){
    int maxIndex = 0;
    for (int i = 1; i < vals.length; i++){
      float newnumber = vals[i];
      if ((newnumber > vals[maxIndex])){
        maxIndex = i;
      }
    }
    return maxIndex;
  }
}
