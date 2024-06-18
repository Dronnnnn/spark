"""Module that covers functionality of ind_temp.py by unit tests"""
# pylint: disable=redefined-outer-name
from datetime import datetime

import pytest

from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType, FloatType,
                               TimestampType, Row, DoubleType, DateType)

from pyspark.testing import assertDataFrameEqual

from ..ind_temp import (filtered_df, agg_by_city, agg_per_day,
                        warm_cities, extended_filtered_df, temp_anomalies)


@pytest.fixture(scope="session")
def spark():
    """
    Pytest fixture for creating a Spark session.
    """
    return (SparkSession.builder
            .appName("PySparkTests")
            .getOrCreate())


@pytest.fixture()
def raw_test_df(spark):
    """
    Pytest fixture for creating a DataFrame with raw test data.
    """
    timezone = "+03:00"
    return spark.createDataFrame([
        Row(source_file="some/dir/City1.csv", temperature_2m='22',
            date=f"2024-06-14 15:00:00{timezone}"),
        Row(source_file="some/dir/City1.csv", temperature_2m='24',
            date=f"2024-06-14 16:00:00{timezone}"),
        Row(source_file="some/dir/City2.csv", temperature_2m='12',
            date=f"2024-06-14 15:00:00{timezone}"),
        Row(source_file="some/dir/City2.csv", temperature_2m='18',
            date=f"2024-06-14 16:00:00{timezone}"),
        Row(source_file="some/dir/City2.csv", temperature_2m='18',
            date=f"2024-06-14 17:00:00{timezone}"),
        Row(source_file="some/dir/City2.csv", temperature_2m='18',
            date=f"2024-06-14 18:00:00{timezone}"),
        Row(source_file="some/dir/City2.csv", temperature_2m='18',
            date=f"2024-06-14 19:00:00{timezone}"),
        Row(source_file="some/dir/City2.csv", temperature_2m='18',
            date=f"2024-06-14 20:00:00{timezone}"),
        Row(source_file="some/dir/City2.csv", temperature_2m='48',
            date=f"2024-06-14 21:00:00{timezone}"),
    ])


@pytest.fixture()
def clear_test_df(spark):
    """
    Pytest fixture for creating a DataFrame with filtered test data.
    """
    schema = StructType([
        StructField("city", StringType(), True),
        StructField("temperature", FloatType(), True),
        StructField("date", TimestampType(), True)
    ])
    return spark.createDataFrame([
        Row(city="City1", temperature=22.0, date=datetime(2024, 6, 14, 15, 0)),
        Row(city="City1", temperature=24.0, date=datetime(2024, 6, 14, 16, 0)),
        Row(city="City2", temperature=12.0, date=datetime(2024, 6, 14, 15, 0)),
        Row(city="City2", temperature=18.0, date=datetime(2024, 6, 14, 16, 0)),
        Row(city="City2", temperature=18.0, date=datetime(2024, 6, 14, 17, 0)),
        Row(city="City2", temperature=18.0, date=datetime(2024, 6, 14, 18, 0)),
        Row(city="City2", temperature=18.0, date=datetime(2024, 6, 14, 19, 0)),
        Row(city="City2", temperature=18.0, date=datetime(2024, 6, 14, 20, 0)),
        Row(city="City2", temperature=48.0, date=datetime(2024, 6, 14, 21, 0)),
    ], schema)


@pytest.fixture()
def agg_test_df(spark):
    """
    Pytest fixture for creating a DataFrame with aggregated test data.
    """
    schema = StructType([
        StructField("city", StringType(), True),
        StructField("avg_temperature", DoubleType(), True),
        StructField("max_temperature", FloatType(), True),
        StructField("min_temperature", FloatType(), True),
    ])
    return spark.createDataFrame([
        Row(city="City1", avg_temperature=23.0, max_temperature=24.0, min_temperature=22.0),
        Row(city="City2", avg_temperature=21.43, max_temperature=48.0, min_temperature=12.0),
    ], schema)


@pytest.fixture()
def agg_test_per_day_df(spark):
    """
    Pytest fixture for creating a DataFrame with aggregated per day test data.
    """
    schema = StructType([
        StructField("city", StringType(), True),
        StructField("date_only", DateType(), True),
        StructField("avg_day_temperature", DoubleType(), True),
        StructField("max_day_temperature", FloatType(), True),
        StructField("min_day_temperature", FloatType(), True),
        StructField("stddev_day_temperature", DoubleType(), True),
    ])
    return spark.createDataFrame([
        Row(city="City1", date_only=datetime(2024, 6, 14),
            avg_day_temperature=23.0, max_day_temperature=24.0, min_day_temperature=22.0,
            stddev_day_temperature=1.41),
        Row(city="City2", date_only=datetime(2024, 6, 14),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
    ], schema)


@pytest.fixture()
def warm_cities_test_df(spark):
    """
    Pytest fixture for creating a DataFrame with warm cities test data.
    """
    schema = StructType([
        StructField("city", StringType(), True),
        StructField("min_temperature", FloatType(), True),
        StructField("max_temperature", FloatType(), True),
    ])
    return spark.createDataFrame([
        Row(city="City1", min_temperature=22.0, max_temperature=24.0, )
    ], schema)


@pytest.fixture()
def extended_filtered_test_df(spark):
    """
    Pytest fixture for creating a DataFrame with extended filtered test data.
    """
    schema = StructType([
        StructField("city", StringType(), True),
        StructField("date_only", DateType(), True),
        StructField("temperature", FloatType(), True),
        StructField("date", TimestampType(), True),
        StructField("avg_day_temperature", DoubleType(), True),
        StructField("max_day_temperature", FloatType(), True),
        StructField("min_day_temperature", FloatType(), True),
        StructField("stddev_day_temperature", DoubleType(), True),
    ])
    return spark.createDataFrame([
        Row(city="City1", date_only=datetime(2024, 6, 14), temperature=22.0,
            date=datetime(2024, 6, 14, 15, 0),
            avg_day_temperature=23.0, max_day_temperature=24.0, min_day_temperature=22.0,
            stddev_day_temperature=1.41),
        Row(city="City1", date_only=datetime(2024, 6, 14), temperature=24.0,
            date=datetime(2024, 6, 14, 16, 0),
            avg_day_temperature=23.0, max_day_temperature=24.0, min_day_temperature=22.0,
            stddev_day_temperature=1.41),
        Row(city="City2", date_only=datetime(2024, 6, 14), temperature=12.0,
            date=datetime(2024, 6, 14, 15, 0),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
        Row(city="City2", date_only=datetime(2024, 6, 14), temperature=18.0,
            date=datetime(2024, 6, 14, 16, 0),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
        Row(city="City2", date_only=datetime(2024, 6, 14), temperature=18.0,
            date=datetime(2024, 6, 14, 17, 0),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
        Row(city="City2", date_only=datetime(2024, 6, 14), temperature=18.0,
            date=datetime(2024, 6, 14, 18, 0),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
        Row(city="City2", date_only=datetime(2024, 6, 14), temperature=18.0,
            date=datetime(2024, 6, 14, 19, 0),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
        Row(city="City2", date_only=datetime(2024, 6, 14), temperature=18.0,
            date=datetime(2024, 6, 14, 20, 0),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
        Row(city="City2", date_only=datetime(2024, 6, 14), temperature=48.0,
            date=datetime(2024, 6, 14, 21, 0),
            avg_day_temperature=21.43, max_day_temperature=48.0, min_day_temperature=12.0,
            stddev_day_temperature=11.93),
    ], schema)


@pytest.fixture()
def anomalies_test_df(spark):
    """
    Pytest fixture for creating a DataFrame with warm cities test data.
    """
    schema = StructType([
        StructField("city", StringType(), True),
        StructField("date", TimestampType(), True),
        StructField("temperature", FloatType(), True),
        StructField("avg_day_temperature", DoubleType(), True),
    ])
    return spark.createDataFrame([
        Row(city="City2", date=datetime(2024, 6, 14, 21, 0), temperature=48.0,
            avg_day_temperature=21.43),
    ], schema)


def test_filtered_df(raw_test_df, clear_test_df):
    """
    Test for the filtered_df function.
    """
    clear_df = filtered_df(raw_test_df)
    assertDataFrameEqual(clear_df, clear_test_df)


def test_agg_by_city(clear_test_df, agg_test_df):
    """
    Test for the agg_by_city function.
    """
    agg_df = agg_by_city(clear_test_df)
    assertDataFrameEqual(agg_df, agg_test_df)


def test_agg_per_day(clear_test_df, agg_test_per_day_df):
    """
    Test for the agg_per_day function.
    """
    agg_per_day_df = agg_per_day(clear_test_df)
    assertDataFrameEqual(agg_per_day_df, agg_test_per_day_df)


def test_warm_cities(clear_test_df, warm_cities_test_df):
    """
    Test for the warm_cities function.
    """
    warm_cities_df = warm_cities(clear_test_df)
    assertDataFrameEqual(warm_cities_df, warm_cities_test_df)


def test_extended_filtered_df(clear_test_df, agg_test_per_day_df, extended_filtered_test_df):
    """
    Test for the extended_filtered_df function.
    """
    extended_df = extended_filtered_df(clear_test_df, agg_test_per_day_df)
    assertDataFrameEqual(extended_df, extended_filtered_test_df)


def test_temp_anomalies(extended_filtered_test_df, anomalies_test_df):
    """
    Test for the temp_anomalies function.
    """
    anomalies_df = temp_anomalies(extended_filtered_test_df)
    assertDataFrameEqual(anomalies_df, anomalies_test_df)
