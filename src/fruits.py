from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

import pandas as pd
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from pyspark.sql.functions import col, pandas_udf, PandasUDFType

from pyspark import SparkContext


sc = SparkContext()


spark = SparkSession \
        .builder \
        .appName("Python Spark create RDD example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
# df = spark.sparkContext\
#         .parallelize([(1, 2, 3, 'a b c'),
#         (4, 5, 6, 'd e f'),
#         (7, 8, 9, 'g h i')])\
#         .toDF(['col1', 'col2', 'col3','col4'])
# df.show()

# Chargement des images
images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("./data")
# display(images.limit(5))
print("OK: read.format")

# Test de chargement du modèle CNN
model = ResNet50(include_top=False)
model.summary()
print("OK: CNN model")

# Test
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])
df.show(truncate=False)
df.printSchema()

# Construction du modème
bc_model_weights = sc.broadcast(model.get_weights())
def model_fn():
    """
    Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
    """
    model = ResNet50(weights=None, include_top=False)
    model.set_weights(bc_model_weights.value)
    return model

# Preprocessing des images
def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

# Extraction des features des images
def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
    
    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                                is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# We can now run featurization on our entire Spark DataFrame.
# NOTE: This can take a long time (about 10 minutes) since it applies a large model to the full dataset.
# features_df = images.repartition(16).select(col("path"), featurize_udf("content").alias("features"))
features_df = images.repartition(16).select(featurize_udf("content").alias("features"))
# features_df.write.mode("overwrite").parquet("dbfs:/ml/tmp/flower_photos_features")

# Tranformation en Vector

features_df.show(5)
print("Avant :")
features_df.printSchema()

# from pyspark.ml.functions import array_to_vector
# features_df = features_df.select(array_to_vector('features')).collect()

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
features_df = features_df.select(
    list_to_vector_udf(features_df["features"]).alias("features")
)
print("Après :")
features_df.printSchema()
print(f"OK: list_to_vector_udf")

# inputCols = ["features"]
# outputCol = "features_vect"
# df_va = VectorAssembler(inputCols = inputCols, outputCol = outputCol)
# features_df = df_va.transform(features_df)
# features_df.printSchema()
# print(f"OK: VectorAssembler")

# PCA
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
# model = pca.fit(df)
model = pca.fit(features_df)

result = model.transform(features_df).select("pcaFeatures")
result.show(truncate=False)

# from pyspark.sql.functions import col
# result.withColumn('pcaFeatures', col('pcaFeatures').cast('string')).write.csv("./rezuzu.csv")
pd = result.toPandas()
pd.to_csv("./rezuzuzu.csv")
