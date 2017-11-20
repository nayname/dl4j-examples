package org.deeplearning4j.sentiment.reviews;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import java.io.File;

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
	SentimentAnalyzer sa = new SentimentAnalyzer();
    if (args.length > 0 && args[0].equals("create")) 
      sa.createModel();      
    
    sa.evaluate(sa.test);	
	
    //After training: load a single example and generate predictions
    File firstPositiveReviewFile = new File(FilenameUtils.concat(sa.DATA_PATH, "aclImdb/test/pos/0_10.txt"));
	
    System.err.println("File load:"+(System.currentTimeMillis() - start));
	sa.fetchReviews(FileUtils.readFileToString(firstPositiveReviewFile));
	System.err.println("All done:"+(System.currentTimeMillis() - start));
  }
}
