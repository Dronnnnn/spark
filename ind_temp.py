"""Module that transforms data of temperature of indian cities"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f

from .const import (
    TEMP_ANOMALY_DATA,
    WARM_CITIES_DATA,
    AGG_DATA_PER_DAY,
    AGG_DATA_BY_CITY,
    CLEAR_DATA,
    CSV_PATH,
    REGEXP_PATTERN,
    DATE_PATTERN
)


def create_spark_session():
    """
    Create and configure a Spark session.

    Returns:
        SparkSession: Configured Spark session.
    """
    spark = (SparkSession.builder.appName("IndCityTemp")
             .config("spark.executor.memory", "8g")
             .config("spark.driver.memory", "8g")
             .config("spark.executor.instances", "4")
             .config("spark.executor.cores", "4")
             .config("spark.memory.fraction", "0.6")
             .config("spark.memory.offHeap.enabled", "true")
             .config("spark.memory.offHeap.size", "4g")
             .getOrCreate())

    return spark


def read_data(spark: SparkSession, filename: str):
    """
    Read data from a CSV file and add a source file column.

    Args:
        spark (SparkSession): Spark session.
        filename (str): Path to the CSV file.

    Returns:
        DataFrame: Raw DataFrame with source file column.
    """
    raw_df = (spark.read.csv(filename, header=True)
              .withColumn("source_file", f.input_file_name()))

    return raw_df


def filtered_df(raw_df: DataFrame):
    """
    Filter and transform the raw DataFrame.

    Args:
        raw_df (DataFrame): Raw DataFrame.

    Returns:
        DataFrame: Filtered and transformed DataFrame.
    """
    ind_cities_df = (raw_df
                     .withColumn("file_name", f.regexp_extract("source_file", REGEXP_PATTERN, 1))
                     .withColumn('temperature', f.col('temperature_2m').cast('float'))
                     .withColumn('date', f.to_timestamp(f.col("date"), DATE_PATTERN))
                     )

    clear_df = (
        ind_cities_df
        .select(
            f.col('file_name').alias('city'),
            f.col('temperature'),
            f.col('date')
        )
        .orderBy('city', 'date')
    )
    clear_df.repartition(200)

    return clear_df


def agg_by_city(clear_df: DataFrame):
    """
    Aggregate data by city.

    Args:
        clear_df (DataFrame): Filtered DataFrame.

    Returns:
        DataFrame: Aggregated DataFrame by city.
    """
    agg_df = (
        clear_df
        .groupBy("city")
        .agg(
            f.round(f.avg("temperature"), 2).alias("avg_temperature"),
            f.round(f.max("temperature"), 2).alias("max_temperature"),
            f.round(f.min("temperature"), 2).alias("min_temperature")
        )
        .orderBy(f.col("city"))
    )

    return agg_df


def agg_per_day(clear_df: DataFrame):
    """
    Aggregate data per day.

    Args:
        clear_df (DataFrame): Filtered DataFrame.

    Returns:
        DataFrame: Aggregated DataFrame per day.
    """
    agg_per_day_df = (
        clear_df.withColumn("date_only", f.to_date(f.col("date")))
        .groupBy("city", "date_only")
        .agg(
            f.round(f.avg("temperature"), 2).alias("avg_day_temperature"),
            f.round(f.max("temperature"), 2).alias("max_day_temperature"),
            f.round(f.min("temperature"), 2).alias("min_day_temperature"),
            f.round(f.stddev("temperature"), 2).alias("stddev_day_temperature")
        )
        .orderBy(f.col("city"), f.col("date_only"))
    )

    return agg_per_day_df


def warm_cities(clear_df: DataFrame):
    """
    Filter warm cities based on temperature range.

    Args:
        clear_df (DataFrame): Filtered DataFrame.

    Returns:
        DataFrame: DataFrame of warm cities.
    """
    warm_weather_filter = (f.col("min_temperature") >= 13) & (f.col("max_temperature") <= 40)

    warm_cities_df = (
        clear_df
        .groupBy("city")
        .agg(
            f.round(f.min("temperature"), 2).alias("min_temperature"),
            f.round(f.max("temperature"), 2).alias("max_temperature")
        )
        .filter(warm_weather_filter)
    )

    return warm_cities_df


def extended_filtered_df(clear_df: DataFrame, agg_per_day_df: DataFrame):
    """
    Extend the filtered DataFrame with aggregated data per day.

    Args:
        clear_df (DataFrame): Filtered DataFrame.
        agg_per_day_df (DataFrame): Aggregated DataFrame per day.

    Returns:
        DataFrame: Extended DataFrame with aggregated data per day.
    """
    ext_clear_df = clear_df.withColumn("date_only", f.to_date(f.col("date")))

    extended_df = ext_clear_df.join(agg_per_day_df, ["city", "date_only"])

    return extended_df


def temp_anomalies(extended_df: DataFrame):
    """
    Identify temperature anomalies.

    Args:
        extended_df (DataFrame): Extended DataFrame with aggregated data per day.

    Returns:
        DataFrame: DataFrame of temperature anomalies.
    """
    double_stdev = 2 * f.col("stddev_day_temperature")

    upper_deviation = (
            f.col("temperature") > f.col("avg_day_temperature") + double_stdev
    )
    lower_deviation = (
            f.col("temperature") < f.col("avg_day_temperature") - double_stdev
    )

    anomalous_deviation = upper_deviation | lower_deviation

    df = extended_df.withColumn("is_anomaly", anomalous_deviation)

    anomalies_df = (
        df
        .filter(f.col("is_anomaly"))
        .select(
            f.col("city"),
            f.col("date"),
            f.col("temperature"),
            f.col("avg_day_temperature"))
        .orderBy(f.col("city"), f.col("date"))
    )

    return anomalies_df


def write_data_to_csv(df_csv: DataFrame, writing_file: str):
    """
    Write DataFrame to a CSV file.

    Args:
        df_csv (DataFrame): DataFrame to write.
        writing_file (str): Path to the CSV file.
    """
    df_csv.write.option('header', 'true').mode("overwrite").csv(writing_file)


def write_data_to_parquet(df_parquet: DataFrame, writing_file: str):
    """
    Write DataFrame to a Parquet file.

    Args:
        df_parquet (DataFrame): DataFrame to write.
        writing_file (str): Path to the Parquet file.
    """
    df_parquet.write.mode("overwrite").parquet(writing_file)


def main():
    """
    Main function to execute the data processing pipeline.
    """
    spark = create_spark_session()

    raw_df = read_data(spark, CSV_PATH)

    clear_df = filtered_df(raw_df)
    write_data_to_parquet(clear_df, CLEAR_DATA)

    agg_df = agg_by_city(clear_df)
    write_data_to_csv(agg_df, AGG_DATA_BY_CITY)

    agg_per_day_df = agg_per_day(clear_df)
    write_data_to_parquet(agg_per_day_df, AGG_DATA_PER_DAY)

    warm_cities_df = warm_cities(clear_df)
    write_data_to_parquet(warm_cities_df, WARM_CITIES_DATA)

    extended_df = extended_filtered_df(clear_df, agg_per_day_df)
    anomalies_df = temp_anomalies(extended_df)
    write_data_to_parquet(anomalies_df, TEMP_ANOMALY_DATA)


if __name__ == '__main__':
    main()
