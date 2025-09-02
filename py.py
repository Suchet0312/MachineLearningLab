from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, trim, lower, expr, levenshtein
)

# Initialize Spark session
spark = SparkSession.builder.appName("RegexNormalization").getOrCreate()

# Load dataset (with header + inferSchema, so you get column names & correct types)
df = spark.read.csv("/content/sample_data/customer.csv", header=True, inferSchema=True)

print("Initial DataFrame:")
df.show()
df.printSchema()

# --- Step 1: Regex-based cleaning/normalization ---
# Get all string columns
string_cols = [field.name for field in df.schema.fields if field.dataType.simpleString() == "string"]

# Clean each string column
for c in string_cols:
    df = df.withColumn(c, trim(col(c)))  # trim spaces
    df = df.withColumn(c, lower(col(c)))  # lowercase
    df = df.withColumn(c, regexp_replace(col(c), r"[^a-z0-9\s]", ""))  # remove special chars
    df = df.withColumn(c, regexp_replace(col(c), r"\s+", " "))  # normalize spaces

# --- Step 2: Create blocking key ---
df = df.withColumn("block_key", expr("substring(name, 1, 1)"))

# --- Step 3: Generate candidate pairs (self-join on block_key) ---8iku88uu
pairs = (
    df.alias("a")
    .join(df.alias("b"), on="block_key")
    .where(expr("a.id < b.id"))  # avoid self-pairs and duplicates
)

# --- Step 4: Compute similarity (Levenshtein distance) ---
pairs = (
    pairs.withColumn("name_lev_dist", levenshtein(expr("a.name"), expr("b.name")))
         .withColumn("city_lev_dist", levenshtein(expr("a.city"), expr("b.city")))
)

# Normalize distances into similarity scores
pairs = (
    pairs.withColumn(
        "name_sim",
        1 - (col("name_lev_dist") / expr("greatest(length(a.name), length(b.name))"))
    )
    .withColumn(
        "city_sim",
        1 - (col("city_lev_dist") / expr("greatest(length(a.city), length(b.city))"))
    )
)

# --- Step 5: Classify linked pairs ---
linked_pairs = pairs.where((col("name_sim") > 0.8) & (col("city_sim") > 0.7))

# Select final columns for output
linked_pairs = linked_pairs.select(
    expr("a.id").alias("id1"),
    expr("a.name").alias("name1"),
    expr("a.city").alias("city1"),
    expr("b.id").alias("id2"),
    expr("b.name").alias("name2"),
    expr("b.city").alias("city2"),
    "name_sim",
    "city_sim"
)

# --- Show results ---
linked_pairs.show(truncate=False)

# Stop Spark session
spark.stop()