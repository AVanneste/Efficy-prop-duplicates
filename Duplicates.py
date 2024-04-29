import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, rank, concat_ws
from fuzzywuzzy import fuzz
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
import os

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

# Create SparkSession
spark = SparkSession.builder \
    .appName("BestNameMatcher") \
    .getOrCreate()

# Define a UDF to calculate similarity score using fuzzywuzzy
def similarity_score(name1, name2):
    return fuzz.ratio(name1, name2)

# Register the UDF with return type IntegerType
similarity_udf = udf(similarity_score, IntegerType())

# Function to process data and find best matches
def process_data(df):
    # Select relevant columns and concatenate address components
    df = df.select(concat_ws(' ', df.F_STREET_FR, df.F_STREET_NUM, df.F_POSTCODE, df.F_CITY_FR).alias("Address"),
                   "NAME", "K_PROPERTY")

    # Cross join the DataFrame with itself to get pairs of names
    cross_joined_df = df.alias("df1").crossJoin(df.alias("df2"))

    # Calculate similarity score for each pair of names using fuzzywuzzy
    scored_df = cross_joined_df.withColumn("score_name", similarity_udf(col("df1.NAME"), col("df2.NAME"))) \
                               .withColumn("score_address", similarity_udf(col("df1.Address"), col("df2.Address")))

    # Filter out the rows where the names are the same
    scored_df = scored_df.filter(scored_df["df1.K_PROPERTY"] != scored_df["df2.K_PROPERTY"])

    # Use window function to rank scores within each group of names
    window_spec = Window.partitionBy("df1.NAME").orderBy(scored_df["score_name"].desc())

    # Rank the scores and keep only the best score for each name pair
    best_matches_df = scored_df.select("df1.K_PROPERTY", "df1.NAME", "df1.Address", 
                                       col("df2.K_PROPERTY").alias("K_PROPERTY2"), 
                                       col("df2.NAME").alias("NAME2"), 
                                       col("df2.Address").alias("Address2"),
                                       "score_name", "score_address",
                                       rank().over(window_spec).alias("rank")) \
                                .filter(col("rank") == 1) \
                                .drop("rank") \
                                .orderBy(col("score_name").desc()) \
                                .filter(col("df1.K_PROPERTY") < col("df2.K_PROPERTY"))

    return best_matches_df

# Streamlit app
def main():
    st.title("Best Name Matcher")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Save the uploaded file locally
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Read the CSV file using Spark
        df = spark.read.csv("temp.csv", header=True, inferSchema=True)
        df = df.sample(False, 10000 / df.count())
        
        # Process data
        processed_df = process_data(df)

        # Display results
        st.dataframe(processed_df.toPandas())

if __name__ == "__main__":
    main()
