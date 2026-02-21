from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("LoanPerformance-SmokeTest")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)

# Lee una partición (ajusta a tu ruta real)
path = "data/parquet/performance/year=2020/quarter=Q4/2020Q4.parquet"
df = spark.read.parquet(path)

print("rows:", df.count())
print("cols:", len(df.columns))
df.select("c001", "c002", "year", "quarter").show(5, truncate=False)

# Ejemplo de agregación (evidencia “in-memory processing”)
agg = (
    df.groupBy("year", "quarter")
      .agg(F.count("*").alias("n_rows"), F.approx_count_distinct("c001").alias("n_loans"))
)

agg.show(truncate=False)

spark.stop()