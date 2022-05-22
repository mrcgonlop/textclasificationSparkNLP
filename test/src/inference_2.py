



from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import sparknlp
import pandas as pd

from pyspark.sql.functions import concat,col,lit, when

from pyspark.sql.types import StructField,StructType,StringType,IntegerType

spark = sparknlp.start()

spark.conf.set("spark.sql.legacy.json.allowEmptyString.enabled", True)
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

schema = StructType([
    StructField("test_index",IntegerType(),True),
    StructField("Text", StringType(), True),
    StructField("class",StructType([
        StructField("annotatorType", StringType(),True),
        StructField("begin" ,IntegerType(),True),
        StructField("end",IntegerType(),True),
        StructField("result",StringType(),True),
        StructField("metadata",StringType(),True),
        StructField("embeddings",StructType([]),True)
    ]),False)
])
#[Row(annotatorType='category', begin=0, end=66, result='Methane (CH4)', metadata={'Ãœxheim': '7.521066E-15', 'Carbon dioxide (CO2)': '1.402646E-10', 'sentence': '0', 'AIR': '2.0145544E-14', 'Nitrogen oxides (NOX)': '4.28333E-8', 'Methane (CH4)': '1.0'}, embeddings=[])]


df = spark.read.csv('out.csv', header=True)
ef = df.select('test_index','Text',"class").toPandas()

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

ef["result"]=ef['class'].apply(lambda x : x[5:-3].split(',')[3])

import numpy as np

conditions = [
    (ef['result'] == " result='Carbon dioxide (CO2)'"),
    (ef['result'] == " result='Methane (CH4)'"),
    (ef['result'] == " result='Nitrogen oxides (NOX)'")]
choices = ['1', '2', '0']
ef['pollutant'] = np.select(conditions, choices, default='-1')
ef['test_index'] = ef['test_index'].astype(int)
ef= ef.sort_values(by=['test_index'])
fds = ef[['test_index','pollutant']]
fds.to_csv("nlp_predictions.csv")
fds.to_json("nlp_predictions.json")

print(fds)





