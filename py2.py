from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import hashlib

# ------------------------------------------------------
# 1. Initialize Spark session
# ------------------------------------------------------
# Stop any existing Spark sessions (avoid conflicts)
try:
    SparkSession.builder.getOrCreate().stop()
except:
    pass  # Ignore if no Spark session is running

spark = SparkSession.builder.appName("L4-RecommendationSystem").getOrCreate()

# ------------------------------------------------------
# 2. Load dataset (JSON file of user-item interactions)
# ------------------------------------------------------
df = spark.read.json("/content/sample_data/movies.json")

# Show first rows and schema
df.show()
df.printSchema()

# Select relevant columns
df = df['user_id', 'product_id', 'score']
df.show()
df.printSchema()

# ------------------------------------------------------
# 3. Convert alphanumeric IDs into integers (hashing)
# ------------------------------------------------------
def AlphanumericToInt(x):
    """
    Hashes alphanumeric strings into numeric IDs using SHA1.
    This is needed because ALS only works with integer IDs for users/items.
    """
    if x is None:
        return None
    return int(hashlib.sha1(x.encode()).hexdigest(), 16) % 10**8

# Register function as a UDF
AlphanumericToInt_udf = udf(AlphanumericToInt, IntegerType())

# Apply hashing to user_id and product_id
id_columns = ["user_id", "product_id"]
for col_name in id_columns:
    new_col_name = f"{col_name}_hashed_int"
    df = df.withColumn(new_col_name, AlphanumericToInt_udf(df[col_name]))

df.show()
df.printSchema()

# ------------------------------------------------------
# 4. Handle Nulls in Hashed ID Columns
# ------------------------------------------------------
# Ensure no null values remain (ALS cannot handle nulls)
df = df.filter(col("user_id_hashed_int").isNotNull() & col("product_id_hashed_int").isNotNull())

# ------------------------------------------------------
# 5. Train-test split
# ------------------------------------------------------
training_data, test_data = df.randomSplit([0.8, 0.2], seed=0)

# ------------------------------------------------------
# 6. Define ALS recommender
# ------------------------------------------------------
als = ALS(
    userCol="user_id_hashed_int",
    itemCol="product_id_hashed_int",
    ratingCol="score",
    coldStartStrategy="drop",  # drop unknowns during prediction
    nonnegative=True           # enforce non-negative predictions
)

# ------------------------------------------------------
# 7. Hyperparameter tuning with ParamGridBuilder
# ------------------------------------------------------
# ALS has several hyperparameters we can tune:
# - rank: number of latent factors
# - maxIter: number of training iterations
# - regParam: regularization parameter (prevents overfitting)
paramGrid = (ParamGridBuilder()
             .addGrid(als.rank, [5, 10])        # try 2 values for rank
             .addGrid(als.maxIter, [5, 10])     # try 2 values for iterations
             .addGrid(als.regParam, [0.01, 0.1])# try 2 values for regularization
             .build())

# ------------------------------------------------------
# 8. Cross-validation setup
# ------------------------------------------------------
# Evaluator: use RMSE (Root Mean Square Error)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="score",
    predictionCol="prediction"
)

# CrossValidator:
# - estimator = ALS model
# - estimatorParamMaps = paramGrid (all combinations of hyperparameters)
# - evaluator = RMSE evaluator
# - numFolds = 3 (split training set into 3 parts for cross-validation)
cv = CrossValidator(
    estimator=als,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

# ------------------------------------------------------
# 9. Train ALS model with Cross-Validation
# ------------------------------------------------------
cvModel = cv.fit(training_data)

# ------------------------------------------------------
# 10. Best model from CV
# ------------------------------------------------------
bestModel = cvModel.bestModel
print("Best Model Parameters:")
print(f" - Rank: {bestModel._java_obj.parent().getRank()}")
print(f" - MaxIter: {bestModel._java_obj.parent().getMaxIter()}")
print(f" - RegParam: {bestModel._java_obj.parent().getRegParam()}")

# ------------------------------------------------------
# 11. Evaluate model performance on test set
# ------------------------------------------------------
predictions = bestModel.transform(test_data)
predictions = predictions.filter(col("prediction").isNotNull())
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error (RMSE) on test data = {rmse}")

# ------------------------------------------------------
# 12. Generate top-5 recommendations for each user
# ------------------------------------------------------
user_recs = bestModel.recommendForAllUsers(5)
user_recs.show(truncate=False)
