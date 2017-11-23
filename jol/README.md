# JOL examples

## Text analyzis

 - feed list of reviews (IMDB) to get santiment analyze. Use previously trained model, or train model and than feed new reviews;
First, create model from conf. 

`MLModel model = new MLModel(conf);`

From each review (text) we create Object, than feed it to model oand label this review as positive or negative.

`MLItem review = new MLItem(text, model);`

> /dl4j-examples/jol/src/main/java/org/deeplearning4j/sentiment/reviews

- suggest
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/basic/BasicRNNExample.java

- data classification (sort animals by class in CSV)
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/BasicCSVClassifier.java

- iris 
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExample.java
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/


- draw plot(??)
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVPlotter.java
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/PlotUtil.java

## Image analyzis

- image classification
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/


## Log analyzis

- UI 
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface/

- log anomaly detection

- ranker

- dating site (text classification + local ranker)

