package org.jol;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.utilities.DataUtilities;
import org.jol.objects.MLConf;
import org.jol.objects.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import com.google.gson.Gson;

public class AnimalsClassifier {

  private static Logger log = LoggerFactory.getLogger(AnimalsClassifier.class);
  
  private static Map<Integer,String> eats = DataUtilities.readEnumCSV("/DataExamples/animals/eats.csv");
  private static Map<Integer,String> sounds = DataUtilities.readEnumCSV("/DataExamples/animals/sounds.csv");
  private static Map<Integer,String> classifiers = DataUtilities.readEnumCSV("/DataExamples/animals/classifiers.csv");

  public static void main(String[] args) throws Exception {
    long start = System.currentTimeMillis();

    MLConf conf = new Gson().fromJson(FileUtils.readFileToString(new File("/home/nayname/dl4j-examples/jol/recources/animals/animals_model_conf.json")),
        MLConf.class);

    DataSet testData = DataUtilities.readCSVDataset("/DataExamples/animals/animals.csv",
        conf.batchSizeTest, conf.labelIndex, conf.numClasses);
    // make the data model for records prior to normalization, because it
    // changes the data.
	
	Map<Integer,Map<String,Object>> animals = makeAnimalsForTesting(testData);
    	
    DataNormalization normalizer = new NormalizerStandardize();
	normalizer.fit(testData);
    normalizer.transform(testData);
	
    
    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    MLModel model = new MLModel(conf);
    
    System.err.println("File load:"+(System.currentTimeMillis() - start));
    
    INDArray output = model.getOutput(testData.getFeatureMatrix());
    
    for (int i = 0; i < output.rows() ; i++) {

      // set the classification from the fitted results
      animals.get(i).put("classifier",
          classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));
    }
    
    System.err.println(animals);
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
  
  /**
   * take the dataset loaded for the matric and make the record model out of it so
   * we can correlate the fitted classifier to the record.
   *
   * @param testData
   * @return
   */
  public static Map<Integer,Map<String,Object>> makeAnimalsForTesting(DataSet testData){
    Map<Integer,Map<String,Object>> animals = new HashMap<>();

    INDArray features = testData.getFeatureMatrix();
    for (int i = 0; i < features.rows() ; i++) {
      INDArray slice = features.slice(i);
      Map<String,Object> animal = new HashMap();

      //set the attributes
      animal.put("yearsLived", slice.getInt(0));
      animal.put("eats", eats.get(slice.getInt(1)));
      animal.put("sounds", sounds.get(slice.getInt(2)));
      animal.put("weight", slice.getFloat(3));

      animals.put(i,animal);
    }
    return animals;

  }
}
