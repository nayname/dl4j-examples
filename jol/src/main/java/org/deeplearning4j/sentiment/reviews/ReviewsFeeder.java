package org.deeplearning4j.sentiment.reviews;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.jol.MLConf;
import org.jol.MLItem;
import org.jol.MLList;
import org.jol.MLModel;

import com.detectum.cache.CacheEntry;
import com.google.gson.Gson;

import java.io.File;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;

/**Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 * This data set contains 25,000 training reviews + 25,000 testing reviews
 *
 * Process:
 * 1. Automatic on first run of example: Download data (movie reviews) + extract
 * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network
 *
 * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
 * additional tuning.
 *
 * NOTE / INSTRUCTIONS:
 * You will have to download the Google News word vector model manually. ~1.5GB
 * The Google News vector model available here: https://code.google.com/p/word2vec/
 * Download the GoogleNews-vectors-negative300.bin.gz file
 * Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black
 */
public class ReviewsFeeder {

  public static void main(String[] args) throws Exception {
    long start = System.currentTimeMillis();

    MLConf conf = new Gson().fromJson(FileUtils.readFileToString(new File("/home/nayname/dl4j-examples/jol/sentiment/sentiment_model_conf.json")),
        MLConf.class);
    
    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    MLModel model = new MLModel(conf);
    
    System.err.println("File load:"+(System.currentTimeMillis() - start));

    File[] files = new File(conf.dataPath+"aclImdb/test/pos/").listFiles();

    HashMap<String, ArrayList<MLItem>> labeled_reviews = new HashMap<>();

    for ( File file : files ) {
      MLItem review = new MLItem(FileUtils.readFileToString(file), model, "dl4j");

      if (!labeled_reviews.containsKey(review.getLabel()))
        labeled_reviews.put(review.getLabel(), new ArrayList<MLItem>());

      labeled_reviews.get(review.getLabel()).add(review);
    }

    if (labeled_reviews.containsKey("positive")) {
      System.err.println("Reviews marked as positive:"+labeled_reviews.get("positive").size());

      for ( MLItem r : labeled_reviews.get("positive")){

      }
    }

    System.err.println("All done:"+(System.currentTimeMillis() - start));
  }
}
