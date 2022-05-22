from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import sparknlp
import pandas as pd

from pyspark.sql.functions import concat,col,lit


spark = sparknlp.start()

spark.conf.set("spark.sql.legacy.json.allowEmptyString.enabled", True)
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

#csv

train1Df = spark.read.csv("./data/train1.csv", header= True).withColumn("Text", concat(col("eprtrSectorName"),lit(": "),col("EPRTRAnnexIMainActivityLabel"))) \
    .select("Text","pollutant")
train2Df = spark.read.csv("./data/train2.csv", header= True, sep=";").withColumn("Text", concat(col("eprtrSectorName"),lit(": "),col("EPRTRAnnexIMainActivityLabel"))) \
    .select("Text","pollutant")

#json
train3Df = spark.createDataFrame(pd.read_json("./data/train3.json")).withColumn("Text", concat(col("eprtrSectorName"),lit(": "),col("EPRTRAnnexIMainActivityLabel"))) \
    .select("Text","pollutant")
train4Df = spark.createDataFrame(pd.read_json("./data/train4.json")).withColumn("Text", concat(col("eprtrSectorName"),lit(": "),col("EPRTRAnnexIMainActivityLabel"))) \
    .select("Text","pollutant")
train5Df = spark.createDataFrame(pd.read_json("./data/train5.json")).withColumn("Text", concat(col("eprtrSectorName"),lit(": "),col("EPRTRAnnexIMainActivityLabel"))) \
    .select("Text","pollutant")

#pdf
#train6Df = spark.read.json("./data/train6out/train6.json").withColumn("Text", concat(col("eprtrSectorName"),lit(": "),col("EPRTRAnnexIMainActivityLabel"))) \
 #   .select("Text","pollutant")

train_pre_split = train1Df.union(train2Df).union(train5Df).union(train3Df).union(train4Df)#.union(train6Df)

(trainingData, testData) = train_pre_split.randomSplit([0.7, 0.3], seed = 100)

trainDF = trainingData.repartition(50)
testDF = testData.repartition(50)

document = DocumentAssembler() \
    .setInputCol("Text") \
    .setOutputCol("document")

#sent_bert_use_cmlm_en_base
bert_sent = BertSentenceEmbeddings.pretrained('sent_small_bert_L8_512') \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
classsifierdl = ClassifierDLApproach() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class") \
    .setLabelColumn("pollutant") \
    .setMaxEpochs(35) \
    .setLr(0.001) \
    .setBatchSize(8) \
    .setEnableOutputLogs(True)

bert_clf_pipeline = Pipeline(
    stages = [
        document,
        bert_sent,
        classsifierdl
    ])

#trainDataset.randomSplit([0.7, 0.3], seed = 100)


bert_pipelineModel = bert_clf_pipeline.fit(trainDF)

preds = bert_pipelineModel.transform(testDF)

preds.select("Text","pollutant","class.result").show(10, truncate=80)

bert_pipelineModel.stages[-1].write().overwrite().save('./classificatos/BertDLClassification')


from sklearn.metrics import classification_report

preds_df = preds.select("Text","pollutant","class.result").toPandas()

preds_df['result'] = preds_df['result'].apply(lambda x : x[0])

print (classification_report(preds_df['result'], preds_df['pollutant']))


"""
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

use = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["sentence"]) \
    .setOutputCol("sentence_embeddings")


document_classifier = ClassifierDLModel.pretrained() \
    .setInputCols(["document", "sentence_embeddings"]) \
    .setOutputCol("class")


pipeline = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    use,
    document_classifier
])
"""