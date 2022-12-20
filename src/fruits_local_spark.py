import sys
import io

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark import SparkContext

import pandas as pd
from PIL import Image
import numpy as np
import time
import datetime
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# spark-submit --deploy-mode cluster s3://oc-ds-p8-fruits-project/fruits.py 16 2
# Devenu inutile :
# spark-submit --deploy-mode cluster --master yarn --driver-memory 5g --executor-memory 5g --num-executors 1 --executor-cores 4 --conf spark.dynamicAllocation.enabled=false s3://oc-ds-p8-fruits-project/fruits.py
# spark-submit --deploy-mode cluster --master yarn --driver-memory 2g --executor-memory 2g --num-executors 1 --executor-cores 1 s3://oc-ds-p8-fruits-project/fruits.py

# Récupération des arguments :
nb_images = int(sys.argv[1])
nb_slaves = int(sys.argv[2]) # Pour capitalisation des résultats

# Démarrage du chrono
t0 = time.time()
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Spark
sc = SparkContext()
spark = SparkSession \
        .builder \
        .appName("Python Spark ResNet50 + PCA") \
        .getOrCreate()
        # .config("spark.some.config.option", "some-value") \

# Chargement des images
images = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load(f"./data_{nb_images}")
print("OK: read.format")
# img_path = images.path.toList()
img_path = list(
    images.select('path').toPandas()['path']
)
print(f"img_path = {img_path}")

# Chargement du modèle CNN
model = ResNet50(include_top=False)
model.summary()
print("OK: CNN model")

# Construction du modèle
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

# Réduire la valeur 1024 si problème de mémoire (Out Of Memory)
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "128")
# print("OK: spark.conf.set")

# Featurisation
# features_df = images.repartition(16).select(featurize_udf("content").alias("features")) # Recommandé de ne pas utiliser .repartition, nécessite des ressources
# features_df = images.repartition(8).select(featurize_udf("content").alias("features")) # Recommandé de ne pas utiliser .repartition, nécessite des ressources
features_df = images.select(featurize_udf("content").alias("features")) # Recommandé de ne pas utiliser .repartition, nécessite des ressources

# Tranformation en Vector
# from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.linalg import Vectors as ml_Vectors
from pyspark.ml.linalg import VectorUDT as ml_VectorUDT
from pyspark.sql.functions import udf
list_to_vector_udf = udf(lambda l: ml_Vectors.dense(l), ml_VectorUDT())
features_df = features_df.select(
    list_to_vector_udf(features_df["features"]).alias("features")
)
print(f"OK: list_to_vector_udf")

# inputCols = ["features"]
# outputCol = "features_vect"
# df_va = VectorAssembler(inputCols = inputCols, outputCol = outputCol)
# features_df = df_va.transform(features_df)
# features_df.printSchema()
# print(f"OK: VectorAssembler")

# PCA
# pca = PCA(k=20, inputCol="features", outputCol="pcaFeatures")
# pca.setOutputCol("pcaFeatures")
# print(f"OK: PCA()")
# model = pca.fit(features_df)
# print(f"OK: pca.fit")
# result = model.transform(features_df).select("pcaFeatures") #.collect()
# result.show(truncate=False) # Supprimé après l'ajoute du .collect()
# print("OK: model.transform")

# PCA avec RDD
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors as mllib_Vectors
features_clc = features_df.select('features').collect()
features_vec = [mllib_Vectors.dense(f[0]) for f in features_clc]
rows = sc.parallelize(features_vec)
mat = RowMatrix(rows)
k_dim = 2
pca = mat.computePrincipalComponents(k_dim)
projected = mat.multiply(pca).rows.collect()

elapsed = time.time() - t0
print(f"OK: mat.multiply")

# Ecriture du résultat de l'ACP
columns = [f"F{k+1}" for k in range(k_dim)]
projected_df = pd.DataFrame(data=projected, columns=columns)
projected_df['Path'] = img_path
print(f"projected_df = {projected_df}")

pca_file = f"./pca_{nb_images}_{nb_slaves}_{timestamp}.csv"
projected_df.to_csv(pca_file, sep=';')
print("OK: projected_df.to_csv S3")

# Ecriture du temps écoulé
print(f"Temps utilisateur : {elapsed}")
exec_df = pd.DataFrame({'nb_images': [nb_images], 
                        'nb_slaves': [nb_slaves],
                        'elapsed': [elapsed]})
report_file = f"./report_{nb_images}_{nb_slaves}_{timestamp}.csv"
exec_df.to_csv(report_file, sep=';')
print("OK: exec_df.to_csv S3")

spark.stop()