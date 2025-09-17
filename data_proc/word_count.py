import argparse
import re
from pyspark.sql import SparkSession, functions as F, types as T

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def main():
    parser = argparse.ArgumentParser(description="Word count in pyspark on Dataproc")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_word_len", type=int, default=1)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("WordCount")
        .getOrCreate()
    )


    df = spark.read.text(args.input).withColumnRenamed("value", "line")

    # 1. Count the number of lines
    line_count = df.count()

    @F.udf(T.ArrayType(T.StringType()))
    def tokenize_udf(s):
        if s is None:
            return []
        return [w.lower() for w in TOKEN_RE.findall(s.lower()) if len(w) >= args.min_word_len]

    tokens = df.select(F.explode(tokenize_udf(F.col("line"))).alias("word"))
    tokens = tokens.filter(F.col("word") != "")

    # Count the number of words (total tokens)
    word_count = tokens.count()

    # Count unique words
    unique_word_count = tokens.select("word").distinct().count()

    # Count the occurrence of each word
    word_counts = tokens.groupBy("word").agg(F.count("*").alias("count"))

    # Top 5 most frequent words
    top5 = word_counts.orderBy(F.desc("count"), F.asc("word")).limit(5)

    # ---- Print summaries to driver log ----
    print("==== Summary ====")
    print(f"Lines: {line_count}")
    print(f"Words: {word_count}")
    print(f"Unique words: {unique_word_count}")
    print("Top 5 words:")
    for row in top5.collect():
        print(f"{row['word']}\t{row['count']}")


    # Save files
    out_base = args.output.rstrip("/")
    metrics_df = spark.createDataFrame(
        [(int(line_count), int(word_count), int(unique_word_count))],
        schema="lines LONG, words LONG, unique_words LONG",
    )
    metrics_df.coalesce(1).write.mode("overwrite").json(f"{out_base}/metrics_json")
    metrics_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_base}/metrics_csv")
    word_counts.orderBy(F.desc("count"), F.asc("word")).coalesce(1).write.mode("overwrite").option("header", True).csv(
        f"{out_base}/word_counts_csv"
    )
    top5.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_base}/top5_csv")
    spark.stop()

if __name__ == "__main__":
    main()
