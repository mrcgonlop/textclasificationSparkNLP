



from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml.pipeline import PipelineModelReader as sparkpipereader
from pyspark.ml.pipeline import PipelineModel as sparkpipe
import sparknlp
import pandas as pd

from pyspark.sql.functions import concat,col,lit, when



spark = sparknlp.start()

spark.conf.set("spark.sql.legacy.json.allowEmptyString.enabled", True)
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

testDF = spark.read.csv("./data/test_x.csv", header= True).withColumn("Text", concat(col("eprtrSectorName"),lit(": "),col("EPRTRAnnexIMainActivityLabel"))) \
    .select("test_index","Text").repartition(10)

#sent_bert_use_cmlm_en_base

document = DocumentAssembler() \
    .setInputCol("Text") \
    .setOutputCol("document")

#sent_bert_use_cmlm_en_base
bert_sent = BertSentenceEmbeddings.pretrained('sent_small_bert_L8_512') \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
classsifierdl = ClassifierDLModel.load("./classificators/BertDLClassification/") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class")

bert_clf_pipeline = Pipeline(
    stages = [
        document,
        bert_sent,
        classsifierdl
    ])

empty_df = spark.createDataFrame([['']]).toDF("Text")

pipelineModel = bert_clf_pipeline.fit(empty_df)
print("finished loading model")
preds = pipelineModel.transform(testDF)\

preds.toPandas().to_csv('out.csv')
