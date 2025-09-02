from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import hashlib

# ------------------------------------------------------
# 1. Initialize Spark session
# ------------------------------------------------------
# Stop any existing Spark sessions
try:
    SparkSession.builder.getOrCreate().stop()
except:
    pass # Ignore if no Spark session is running

spark = SparkSession.builder.appName("L4").getOrCreate()

# ------------------------------------------------------
# 2. Load dataset (JSON file of user-item interactions)
# ------------------------------------------------------
df = spark.read.json("/content/sample_data/movies.json")
df.show()
df.printSchema()

# Select relevant columns: user_id, product_id, and score
df = df['user_id', 'product_id', 'score']
df.show()
df.printSchema()

# ------------------------------------------------------
# 3. Convert alphanumeric IDs into integers (hashing)
# ------------------------------------------------------
# Define a function that hashes alphanumeric strings into numeric IDs
def AlphanumericToInt(x):
    if x is None:
        return None
    return int(hashlib.sha1(x.encode()).hexdigest(), 16) % 10**8

# Register function as a UDF (User Defined Function)
AlphanumericToInt_udf = udf(AlphanumericToInt, IntegerType())

# Apply hashing to user_id and product_id columns
id_columns = ["user_id", "product_id"]
for col_name in id_columns:
    new_col_name = f"{col_name}_hashed_int"
    df = df.withColumn(new_col_name, AlphanumericToInt_udf(df[col_name]))

df.show()
df.printSchema()

# ------------------------------------------------------
# 4. Handle Nulls in Hashed ID Columns
# ------------------------------------------------------
df = df.filter(col("user_id_hashed_int").isNotNull() & col("product_id_hashed_int").isNotNull())

# ------------------------------------------------------
# 5. Train-test split
# ------------------------------------------------------
# Split dataset into training (80%) and testing (20%)
training_data, test_data = df.randomSplit([0.8, 0.2], seed=0)

# ------------------------------------------------------
# 6. Define ALS (Alternating Least Squares) recommender
# ------------------------------------------------------
als = ALS(
    userCol="user_id_hashed_int",
    itemCol="product_id_hashed_int",
    ratingCol="score",
    coldStartStrategy="drop",  # drop rows where prediction cannot be made
    nonnegative=True           # enforce non-negative predictions
)

# ------------------------------------------------------
# 7. Train ALS model on training data
# ------------------------------------------------------
model = als.fit(training_data)

# ------------------------------------------------------
# 8. Generate predictions on test data
# ------------------------------------------------------
predictions = model.transform(test_data)
predictions = predictions.filter(col("prediction").isNotNull())
predictions.show()

# ------------------------------------------------------
# 9. Evaluate model performance (RMSE)
# ------------------------------------------------------
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="score",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error (RMSE) on test data = {rmse}")

# ------------------------------------------------------
# 10. Generate top-5 recommendations for each user
# ------------------------------------------------------
user_recs = model.recommendForAllUsers(5)
user_recs.show(truncate=False)