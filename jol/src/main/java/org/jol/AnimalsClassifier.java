package org.jol;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.utilities.DataUtilities;
import org.jol.objects.MLConf;
import org.jol.objects.MLItem;
import org.jol.objects.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.gson.Gson;

public class AnimalsClassifier {

  private static Logger log = LoggerFactory.getLogger(AnimalsClassifier.class);

  public static void main(String[] args) throws Exception {
    long start = System.currentTimeMillis();

    MLConf conf = new Gson().fromJson(FileUtils.readFileToString(new File("/home/nayname/dl4j-examples/jol/recources/animals/animals_model_conf.json")),
        MLConf.class);

    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    MLModel model = new MLModel(conf);

    DataSet testData = DataUtilities.readCSVDataset("/DataExamples/animals/animals.csv",
        conf.batchSizeTest, conf.labelIndex, conf.numClasses);

    System.err.println("File load:"+(System.currentTimeMillis() - start));

    Map<String, ArrayList<MLItem>> animals = new HashMap<String,ArrayList<MLItem>>();

    INDArray features = testData.getFeatureMatrix();
    for (int i = 0; i < features.rows() ; i++) {
      INDArray slice = features.slice(i);

      MLItem animal = new MLItem(slice, model);

      String label = animal.getLabel();

      if (!animals.containsKey(label))
        animals.put(label, new ArrayList<>());

      animals.get(label).add(animal);
    }

    System.err.println(animals);
  }
}
