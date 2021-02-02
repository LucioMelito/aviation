# Databricks notebook source
import datetime
import glob
import pandas as pd

from pyspark.sql import functions as fn
from pyspark.sql.functions import col, lit, pandas_udf, PandasUDFType
from pyspark.sql import Window as W
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Load look-up tables and pre-process FR24 data

# COMMAND ----------

fr_path = '/mnt/AviationDataAnalysis/Flight_Data_Files_Raw/'
ref_path = '/mnt/AviationDataAnalysis/'
out_path = '/dbfs/mnt/AviationDataAnalysis/Output/'

start_date = '2020-01-01'
current_date = datetime.datetime.now().strftime('%Y-%m-%d')

# COMMAND ----------

# define FR24 data schema to avoid reading all files
fr_schema = StructType([StructField('address', StringType(), True),
                        StructField('altitude', IntegerType(), True),
                        StructField('callsign', StringType(), True),
                        StructField('date', DateType(), True),
                        StructField('destination_iata', StringType(), True),
                        StructField('destination_icao', StringType(), True),
                        StructField('equipment', StringType(), True),
                        StructField('event', StringType(), True),
                        StructField('fligth', StringType(), True),
                        StructField('flight_id', IntegerType(), True),
                        StructField('latitude', DoubleType(), True),
                        StructField('longitude', DoubleType(), True),
                        StructField('operator', StringType(), True),
                        StructField('origin_iata', StringType(), True),
                        StructField('origin_icao', StringType(), True),
                        StructField('registration', StringType(), True),
                        StructField('time', StringType(), True)])

# COMMAND ----------

# load FR data and filter 'landed' events
df = (spark
      .read
      .option("header", "true")
      .option("delimiter", ",")
      .schema(fr_schema)
      .csv(fr_path + '*.csv'))

# COMMAND ----------

df_codes = ((spark
             .read
             .option("header", "true")
             .option("inferSchema", "true")
             .option("delimiter", ",")
             .csv(ref_path + 'airport-codes_csv2.csv'))
             .select(['iso_country', 'iata_code', 'coordinates'])
             .dropDuplicates())

split_col = fn.split(df_codes['coordinates'], ', ')
df_codes_1 = (df_codes
              .withColumn('longitude', split_col.getItem(0).cast('double'))
              .withColumn('latitude', split_col.getItem(1).cast('double')))

# COMMAND ----------

df_reg = pd.read_csv('/dbfs' + ref_path + 'registration_file.csv', parse_dates=['first_date', 'last_date'])

# COMMAND ----------

role_dict = {1.0: 'cargo',
             2.0: 'passenger'}

vals = ['', 'A', 'B', 'C', 'D']

cols = ['registration', 'equipment', 'operator']
cols1 = ['first_date', 'last_date', 'Role', 'Seats', 'Payload 1', 'Payload 2', 'Payload 3']
cols2 = ['Start {}', 'End {}', 'Role - {}', 'Seats - {}', 'Payload 1 - {}', 'Payload 2 - {}', 'Payload 3 - {}']

# COMMAND ----------

def process_date(x):
  if x == 'NOW':
    return datetime.datetime.now().date()
  elif type(x) is float or type(x) is int or type(x) is str:
    try:
      x = int(x)
      return datetime.fromordinal(datetime(1900, 1, 1).toordinal() + x - 2)
    except:
      return None
  else:
    return x

def process_reg(df):
  df = df[df['Include'] == 1]
  
  for i in range(1, 4):
    end = vals[i]
    start = vals[i+1]
    idx = df[f'Start {start}'] == df[f'End {end}']
    df.loc[idx, f'Start {start}'] = df.loc[idx, f'Start {start}'].map(lambda x: int(x) + 1 if (type(x) is float or type(x) is str or type(x) is int) and x != 'NOW' and x != '-' else x)
  
  tmp = df[cols + cols1].dropna(subset=['Role'])
  for val in vals[1:]:
    tmp1 = df[cols + [x.format(val) for x in cols2]]
    tmp1 = tmp1.rename(columns={k.format(val):v for k, v in zip(cols2, cols1)}).dropna(subset=['Role'])
    tmp = pd.concat([tmp, tmp1], sort=True)
    
  df = tmp
  df['Role'] = df['Role'].map(role_dict)
  df['first_date'] = df['first_date'].apply(process_date)
  df['last_date'] = df['last_date'].apply(process_date)
  df = df.dropna(subset=['first_date', 'last_date'])
  
  tmp1 = df[cols + ['first_date'] + cols1[2:]].rename(columns={'first_date': 'reg_date'})
  tmp2 = df[cols + ['last_date'] + cols1[2:]].rename(columns={'last_date': 'reg_date'})
  df = pd.concat([tmp1, tmp2], sort=True)
  
  return df

# COMMAND ----------

df_reg = spark.createDataFrame(process_reg(df_reg))

# COMMAND ----------

df_countries = ((spark
                 .read
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .option("delimiter", ",")
                 .csv(ref_path + 'Country_ref.csv'))
                .drop('Region')
                .withColumnRenamed('Region1', 'region')
                .withColumnRenamed('Country', 'country')
                .withColumnRenamed('Alpha-3 code', 'iso3'))

# COMMAND ----------

# join airport- and country-data
df_codes_2 = (df_codes_1
              .join(df_countries.select(['country', 'iso_country', 'region', 'iso3']), on='iso_country')
              .dropDuplicates())

# COMMAND ----------

df_distance = (spark
               .read
               .option("header", "true")
               .option("inferSchema", "true")
               .option("delimiter", ",")
               .csv(ref_path + 'iata_distance.csv'))

# COMMAND ----------

df_priority = df.withColumn('priority', (fn.when(col('event') == 'landed', 0)
                                            .otherwise(fn.when(col('event') == 'descent', 1)
                                                       .otherwise(fn.when(col('event') == 'cruising', 2)
                                                                  .otherwise(fn.when(col('event') == 'takeoff', 3)
                                                                             .otherwise(4))))))

# COMMAND ----------

w = W.partitionBy('flight_id', 'registration')

df_filtered = (df_priority
               .withColumn('min_p', fn.min('priority').over(w))
               .where(col('priority') == col('min_p')))

# COMMAND ----------

def join_and_filter(df, reg):
  w = W.partitionBy('flight_id', 'registration')
  
  df_1 = (df
          .join(reg, on=['registration', 'equipment', 'operator'], how='inner')
          .withColumnRenamed('Seats', 'n')
          .withColumnRenamed('Role', 'passenger')
          .withColumn('time_diff', fn.unix_timestamp(col('date')) - fn.unix_timestamp(col('reg_date'))))
  
  df_2 = (df_1
          .withColumn('min_time_diff', fn.min(fn.abs(col('time_diff'))).over(w))
          .where(fn.abs(col('time_diff')) == col('min_time_diff'))
          .dropDuplicates(subset=['flight_id', 'registration', 'passenger', 'n', 'Payload 1']))
  
  return df_2

# COMMAND ----------

flights_landed = join_and_filter(df_filtered, df_reg)

# COMMAND ----------

# select payload value
flights_landed_2 = (flights_landed
                    .join(df_codes_2.select('iata_code', 'iso_country', 'region'), on=flights_landed.origin_iata == df_codes_2.iata_code, how='inner')
                    .withColumnRenamed('iso_country', 'origin_iso_country')
                    .withColumnRenamed('region', 'origin_region')
                    .drop('iata_code')
                    .join(df_codes_2.select('iata_code', 'iso_country', 'region'), on=flights_landed.destination_iata == df_codes_2.iata_code, how='inner')
                    .withColumnRenamed('iso_country', 'destination_iso_country')
                    .withColumnRenamed('region', 'destination_region')
                    .drop('iata_code')
                    .withColumn('type', (fn.when(col('origin_iso_country') == col('destination_iso_country'), 'domestic')
                                           .otherwise(fn.when(col('origin_region') == col('destination_region'), 'intra-regional')
                                                        .otherwise('inter-regional'))))
                    .withColumn('payload', (fn.when(col('type') == 'inter-regional', col('Payload 1'))
                                              .otherwise(fn.when(col('type') == 'intra-regional', col('Payload 2'))
                                                           .otherwise(col('Payload 3'))))))

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Airport-level OD aggregations

# COMMAND ----------

od_flows = (flights_landed_2
            .groupBy('date', 'passenger', 'origin_iata', 'destination_iata')
            .agg(fn.count(lit(0)).alias('flights_number'),
                 fn.sum('n').alias('passengers_number'),
                 fn.sum('payload').alias('payload'))
            .filter("origin_iata is not null")
            .filter("destination_iata is not null"))

# COMMAND ----------

# join with distance dataframe and compute ASK metric
od_flows_2 = (od_flows
              .join(df_distance, on=['origin_iata', 'destination_iata'])
              .withColumn('av_seat_km', col('passengers_number')*col('distance')))

# COMMAND ----------

od_flows_3 = (od_flows_2.join(df_codes_2, on=od_flows_2.origin_iata == df_codes.iata_code, how='inner')
              .withColumnRenamed('iso_country', 'origin_iso_country')
              .withColumnRenamed('iso3', 'origin_iso3')
              .withColumnRenamed('country', 'origin_country')
              .withColumnRenamed('region', 'origin_region')
              .withColumnRenamed('latitude', 'origin_latitude')
              .withColumnRenamed('longitude', 'origin_longitude')
              .drop('iata_code', 'coordinates'))

od_flows_4 = (od_flows_3.join(df_codes_2, on=od_flows_3.destination_iata == df_codes.iata_code, how='inner')
              .withColumnRenamed('iso_country', 'destination_iso_country')
              .withColumnRenamed('iso3', 'destination_iso3')
              .withColumnRenamed('country', 'destination_country')
              .withColumnRenamed('region', 'destination_region')
              .withColumnRenamed('latitude', 'destination_latitude')
              .withColumnRenamed('longitude', 'destination_longitude')
              .drop('iata_code', 'coordinates'))

# COMMAND ----------

od_flows_5 = (od_flows_4
              .withColumn('type', (fn.when(col('origin_iso_country') == col('destination_iso_country'), 'domestic')
                                   .otherwise(fn.when(col('origin_region') == col('destination_region'), 'intra-regional')
                                             .otherwise('inter-regional'))))
              .withColumn('day_of_year_old', fn.dayofyear(col('date')))
              .withColumn('year', fn.year(col('date')))
              .withColumn('day_of_years', 365*col('year')+col('day_of_year_old')+fn.floor((col('year')-1)/4))
              .withColumn('week_of_year', ((col('day_of_years')+4)/7).cast('int'))
              .withColumn('day_of_year', fn.when(col('day_of_year_old')==366, 365).otherwise(col('day_of_year_old')))
              .drop('day_of_year_old'))

od_flows_5.cache()
od_flows_5.count()

# COMMAND ----------

# compute dates of weeks for airline-level aggregations
week_dates = (od_flows_5
              .groupby('week_of_year')
              .agg(fn.min('date').alias('week_date'),
                   fn.countDistinct('date').alias('ndays')))

week_dates.cache()
week_dates.count()

# COMMAND ----------

def generate_lag_fn(schema, cols, current_date=current_date):
  # establish whether data daily or weekly 
  date_col = 'date' if 'date' in schema.fieldNames() else 'week_date'
  year_lag = 365 if date_col == 'date' else 364
  
  if date_col == 'date':
    schema_df = StructType(schema.fields +
                           [StructField('flights_number_7days', LongType()),
                            StructField('passengers_number_7days', DoubleType()),
                            StructField('payload_7days', DoubleType()),
                            StructField('av_seat_km_7days', DoubleType()),
                            StructField('lastyr_flights_number_7days', LongType()),
                            StructField('lastyr_passengers_number_7days', DoubleType()),
                            StructField('lastyr_payload_7days', DoubleType()),
                            StructField('lastyr_av_seat_km_7days', DoubleType()),
                            StructField('lastwk_flights_number_7days', DoubleType()),
                            StructField('lastwk_passengers_number_7days', DoubleType()),
                            StructField('lastwk_payload_7days', DoubleType()),
                            StructField('lastwk_av_seat_km_7days', DoubleType())])
  else:
    schema_df = StructType(schema.fields +
                         [StructField('lastyr_flights_number_7days', LongType()),
                          StructField('lastyr_passengers_number_7days', DoubleType()),
                          StructField('lastyr_payload_7days', DoubleType()),
                          StructField('lastyr_av_seat_km_7days', DoubleType()),
                          StructField('lastwk_flights_number_7days', DoubleType()),
                          StructField('lastwk_passengers_number_7days', DoubleType()),
                          StructField('lastwk_payload_7days', DoubleType()),
                          StructField('lastwk_av_seat_km_7days', DoubleType())])
  
  @pandas_udf(schema_df, PandasUDFType.GROUPED_MAP)
  def lag_values(df):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    
    if date_col == 'date':
      min_date, max_date = df[date_col].min(), df[date_col].max()
      index = pd.date_range(min_date, max_date + datetime.timedelta(days=365))
      df = df.set_index(date_col)
      df = df.reindex(index)
      df.index.name = date_col
      
      # fill NAs and compute moving sum
      df[['flights_number', 'passengers_number', 'payload', 'av_seat_km']].fillna(0, inplace=True)
      df['flights_number_7days'] = df['flights_number'].rolling('7D').sum()
      df['passengers_number_7days'] = df['passengers_number'].rolling('7D').sum()
      df['payload_7days'] = df['payload'].rolling('7D').sum()
      df['av_seat_km_7days'] = df['av_seat_km'].rolling('7D').sum()
    else:
      df = df.set_index(date_col)

    # lag values to compute yoy and wow changes
    df['lastyr_flights_number_7days'] = df['flights_number_7days'].shift(year_lag, freq='D')
    df['lastyr_passengers_number_7days'] = df['passengers_number_7days'].shift(year_lag, freq='D')
    df['lastyr_payload_7days'] = df['payload_7days'].shift(year_lag, freq='D')
    df['lastyr_av_seat_km_7days'] = df['av_seat_km_7days'].shift(year_lag, freq='D')

    df['lastwk_flights_number_7days'] = df['flights_number_7days'].shift(7, freq='D')
    df['lastwk_passengers_number_7days'] = df['passengers_number_7days'].shift(7, freq='D')
    df['lastwk_payload_7days'] = df['payload_7days'].shift(7, freq='D')
    df['lastwk_av_seat_km_7days'] = df['av_seat_km_7days'].shift(7, freq='D')

    #drop "empty" rows and clip max_date
    df = df[~((df['flights_number_7days'] == 0)&(df['lastyr_flights_number_7days'] == 0)&(df['lastwk_flights_number_7days'] == 0))]
    df = df.loc[df.index < current_date]

    df[cols] = df[cols].ffill().bfill()

    df = df.reset_index(level=date_col)
    return df
  
  return lag_values

# COMMAND ----------

cols = ['origin_iata', 'destination_iata', 'passenger', 'type',
        'origin_country', 'origin_iso_country', 'origin_region', 'origin_iso3', 'origin_longitude', 'origin_latitude',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3', 'destination_longitude', 'destination_latitude',
        'year', 'week_of_year', 'day_of_year', 'day_of_years']
lag_fn = generate_lag_fn(od_flows_5.schema, cols)

for c in od_flows_5.schema.fieldNames():
  od_flows_5.schema[c].nullable = True
od_flows_yoy = od_flows_5.groupby('origin_iata', 'destination_iata', 'passenger').apply(lag_fn)

# COMMAND ----------

od_flows_wow = (od_flows_yoy
                .withColumn('flights_number_7days_wow_change', 
                            (col('flights_number_7days') - col('lastwk_flights_number_7days'))/col('lastwk_flights_number_7days'))
                .withColumn('passengers_number_7days_wow_change', 
                            (col('passengers_number_7days') - col('lastwk_passengers_number_7days'))/col('lastwk_passengers_number_7days'))
                .withColumn('payload_7days_wow_change',
                              (col('payload_7days') - col('lastwk_payload_7days'))/col('lastwk_payload_7days'))
                .withColumn('av_seat_km_7days_wow_change',
                            (col('av_seat_km_7days') - col('lastwk_av_seat_km_7days'))/col('lastwk_av_seat_km_7days'))
                .withColumn('flights_number_7days_yoy_change', 
                            (col('flights_number_7days') - col('lastyr_flights_number_7days'))/col('lastyr_flights_number_7days'))
                .withColumn('passengers_number_7days_yoy_change',
                            (col('passengers_number_7days') - col('lastyr_passengers_number_7days'))/col('lastyr_passengers_number_7days'))
                .withColumn('payload_7days_yoy_change',
                              (col('payload_7days') - col('lastyr_payload_7days'))/col('lastyr_payload_7days'))
                .withColumn('av_seat_km_7days_yoy_change',
                            (col('av_seat_km_7days') - col('lastyr_av_seat_km_7days'))/col('lastyr_av_seat_km_7days')))

od_flows_wow.cache()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS aviation

# COMMAND ----------

def write_csv(df, folder_path, fname):
  (df
   .coalesce(1)
   .write
   .option("header","true")
   .option("sep",",")
   .mode("overwrite")
   .csv(folder_path + fname))
   
  dbutils.fs.mv(glob.glob('/dbfs' + folder_path + fname + '/*.csv')[0].replace('/dbfs', ''), folder_path + fname + '.csv')
  dbutils.fs.rm(folder_path + fname, True)

# COMMAND ----------

#write_csv(df=od_flows_wow.filter(col('date') >= start_date),
#          folder_path = '/mnt/AviationDataAnalysis/Output/',
#          fname = 'airports_od')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Airport-level OD weekly aggregations

# COMMAND ----------

od_flows_weekly = (od_flows_wow
                   .groupby('origin_iata', 'destination_iata', 'passenger', 'week_of_year')
                   .agg(fn.sum(col('flights_number')).alias('flights_number_7days'),
                        fn.sum(col('passengers_number')).alias('passengers_number_7days'),
                        fn.sum(col('payload')).alias('payload_7days'),
                        fn.sum(col('av_seat_km')).alias('av_seat_km_7days'),
                        fn.first(col('type')).alias('type'),
                        fn.first(col('origin_country')).alias('origin_country'),
                        fn.first(col('origin_iso_country')).alias('origin_iso_country'),
                        fn.first(col('origin_region')).alias('origin_region'),
                        fn.first(col('origin_iso3')).alias('origin_iso3'),
                        fn.first(col('origin_latitude')).alias('origin_latitude'),
                        fn.first(col('origin_longitude')).alias('origin_longitude'),
                        fn.first(col('destination_country')).alias('destination_country'),
                        fn.first(col('destination_iso_country')).alias('destination_iso_country'),
                        fn.first(col('destination_region')).alias('destination_region'),
                        fn.first(col('destination_iso3')).alias('destination_iso3'),
                        fn.first(col('destination_latitude')).alias('destination_latitude'),
                        fn.first(col('destination_longitude')).alias('destination_longitude'))
                   .join(week_dates, on=['week_of_year'])
                   .withColumn('flights_number_avg', col('flights_number_7days')/col('ndays'))
                   .withColumn('passengers_number_avg', col('passengers_number_7days')/col('ndays'))
                   .withColumn('payload_avg', col('payload_7days')/col('ndays'))
                   .withColumn('av_seat_km_avg', col('av_seat_km_7days')/col('ndays'))
                   .drop('flights_number_7days', 'passengers_number_7days', 'payload_7days', 'av_seat_km_7days')
                   .withColumnRenamed('flights_number_avg', 'flights_number_7days')
                   .withColumnRenamed('passengers_number_avg', 'passengers_number_7days')
                   .withColumnRenamed('payload_avg', 'payload_7days')
                   .withColumnRenamed('av_seat_km_avg', 'av_seat_km_7days'))

# COMMAND ----------

for c in od_flows_weekly.schema.fieldNames():
  od_flows_weekly.schema[c].nullable = True
  
cols = ['origin_iata', 'destination_iata', 'passenger', 'type',
        'origin_country', 'origin_iso_country', 'origin_region', 'origin_iso3', 'origin_longitude', 'origin_latitude',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3', 'destination_longitude', 'destination_latitude',
        'week_of_year']

lag_fn = generate_lag_fn(od_flows_weekly.schema, cols)
od_flows_weekly_yoy = od_flows_weekly.groupby('origin_iata', 'destination_iata', 'passenger').apply(lag_fn)

# COMMAND ----------

od_flows_weekly_wow = (od_flows_weekly_yoy
                       .withColumn('flights_number_7days_wow_change', 
                                   (col('flights_number_7days') - col('lastwk_flights_number_7days'))/col('lastwk_flights_number_7days'))
                       .withColumn('passengers_number_7days_wow_change', 
                                   (col('passengers_number_7days') - col('lastwk_passengers_number_7days'))/col('lastwk_passengers_number_7days'))
                       .withColumn('payload_7days_wow_change',
                                   (col('payload_7days') - col('lastwk_payload_7days'))/col('lastwk_payload_7days'))
                       .withColumn('av_seat_km_7days_wow_change',
                                   (col('av_seat_km_7days') - col('lastwk_av_seat_km_7days'))/col('lastwk_av_seat_km_7days'))
                       .withColumn('flights_number_7days_yoy_change', 
                                   (col('flights_number_7days') - col('lastyr_flights_number_7days'))/col('lastyr_flights_number_7days'))
                       .withColumn('passengers_number_7days_yoy_change',
                                   (col('passengers_number_7days') - col('lastyr_passengers_number_7days'))/col('lastyr_passengers_number_7days'))
                       .withColumn('payload_7days_yoy_change',
                                   (col('payload_7days') - col('lastyr_payload_7days'))/col('lastyr_payload_7days'))
                       .withColumn('av_seat_km_7days_yoy_change',
                                   (col('av_seat_km_7days') - col('lastyr_av_seat_km_7days'))/col('lastyr_av_seat_km_7days')))

# COMMAND ----------

#od_flows_weekly_wow.filter("week_date >= '20191201'").write.mode("overwrite").saveAsTable("aviation.airports_od_weekly")
(od_flows_weekly_wow
 .filter(col('week_date') >= start_date)
 .toPandas().to_csv(out_path + 'airports_od_weekly.csv', index=False))

od_flows_wow.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Airport-level aggregations

# COMMAND ----------

airports = (od_flows_5
            .groupby('destination_iata', 'type', 'passenger', 'date')
            .agg(fn.sum(col('flights_number')).alias('flights_number'),
                 fn.sum(col('passengers_number')).alias('passengers_number'),
                 fn.sum(col('payload')).alias('payload'),
                 fn.sum(col('av_seat_km')).alias('av_seat_km'),
                 fn.first(col('destination_country')).alias('destination_country'),
                 fn.first(col('destination_iso_country')).alias('destination_iso_country'),
                 fn.first(col('destination_region')).alias('destination_region'),
                 fn.first(col('destination_iso3')).alias('destination_iso3')))

# COMMAND ----------

airports.schema['type'].nullable = True

cols = ['destination_iata', 'passenger', 'type', 'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3']
lag_fn = generate_lag_fn(airports.schema, cols)
airports_yoy = airports.groupby('destination_iata', 'passenger', 'type').apply(lag_fn)

# COMMAND ----------

(airports_yoy
 .filter(col('date') >= start_date)
 .toPandas().to_csv(out_path + 'airports.csv', index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Country-level OD aggregations

# COMMAND ----------

countries_od = (od_flows_5
                .groupby('origin_country', 'destination_country', 'type', 'passenger', 'date', 'week_of_year')
                .agg(fn.sum(col('flights_number')).alias('flights_number'),
                     fn.sum(col('passengers_number')).alias('passengers_number'),
                     fn.sum(col('payload')).alias('payload'),
                     fn.sum(col('av_seat_km')).alias('av_seat_km'),
                     fn.first(col('origin_iso_country')).alias('origin_iso_country'),
                     fn.first(col('destination_iso_country')).alias('destination_iso_country'),
                     fn.first(col('origin_region')).alias('origin_region'),
                     fn.first(col('destination_region')).alias('destination_region'),
                     fn.first(col('origin_iso3')).alias('origin_iso3'),
                     fn.first(col('destination_iso3')).alias('destination_iso3')))

# COMMAND ----------

for c in countries_od.schema.fieldNames():
  countries_od.schema[c].nullable = True

cols = ['passenger', 'type', 'week_of_year',
        'origin_country', 'origin_iso_country', 'origin_region', 'origin_iso3',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3']

lag_fn = generate_lag_fn(countries_od.schema, cols)
countries_od_yoy = countries_od.groupby('origin_country', 'destination_country', 'passenger').apply(lag_fn)

countries_od_yoy.cache()

# COMMAND ----------

countries_od_wow = (countries_od_yoy
                    .withColumn('flights_number_7days_wow_change', 
                                   (col('flights_number_7days') - col('lastwk_flights_number_7days'))/col('lastwk_flights_number_7days'))
                    .withColumn('passengers_number_7days_wow_change', 
                                   (col('passengers_number_7days') - col('lastwk_passengers_number_7days'))/col('lastwk_passengers_number_7days'))
                    .withColumn('payload_7days_wow_change',
                                   (col('payload_7days') - col('lastwk_payload_7days'))/col('lastwk_payload_7days'))
                    .withColumn('av_seat_km_7days_wow_change',
                                   (col('av_seat_km_7days') - col('lastwk_av_seat_km_7days'))/col('lastwk_av_seat_km_7days'))
                    .withColumn('flights_number_7days_yoy_change', 
                                   (col('flights_number_7days') - col('lastyr_flights_number_7days'))/col('lastyr_flights_number_7days'))
                    .withColumn('passengers_number_7days_yoy_change',
                                   (col('passengers_number_7days') - col('lastyr_passengers_number_7days'))/col('lastyr_passengers_number_7days'))
                    .withColumn('payload_7days_yoy_change',
                                   (col('payload_7days') - col('lastyr_payload_7days'))/col('lastyr_payload_7days'))
                    .withColumn('av_seat_km_7days_yoy_change',
                                   (col('av_seat_km_7days') - col('lastyr_av_seat_km_7days'))/col('lastyr_av_seat_km_7days')))

# COMMAND ----------

#countries_od_wow.filter("date >= '20191201'").write.mode("overwrite").saveAsTable("aviation.countries_od")
(countries_od_wow
 .filter(col('date') >= start_date)
 .toPandas().to_csv(out_path + 'countries_od.csv', index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Country-level OD weekly aggregations

# COMMAND ----------

countries_od_weekly = (countries_od_wow
                       .groupby('origin_country', 'destination_country', 'passenger', 'week_of_year')
                       .agg(fn.sum(col('flights_number')).alias('flights_number_7days'),
                            fn.sum(col('passengers_number')).alias('passengers_number_7days'),
                            fn.sum(col('payload')).alias('payload_7days'),
                            fn.sum(col('av_seat_km')).alias('av_seat_km_7days'),
                            fn.first(col('type')).alias('type'),
                            fn.first(col('origin_iso_country')).alias('origin_iso_country'),
                            fn.first(col('origin_region')).alias('origin_region'),
                            fn.first(col('origin_iso3')).alias('origin_iso3'),
                            fn.first(col('destination_iso_country')).alias('destination_iso_country'),
                            fn.first(col('destination_region')).alias('destination_region'),
                            fn.first(col('destination_iso3')).alias('destination_iso3'))
                       .join(week_dates, on=['week_of_year'])
                       .withColumn('flights_number_avg', col('flights_number_7days')/col('ndays'))
                       .withColumn('passengers_number_avg', col('passengers_number_7days')/col('ndays'))
                       .withColumn('payload_avg', col('payload_7days')/col('ndays'))
                       .withColumn('av_seat_km_avg', col('av_seat_km_7days')/col('ndays'))
                       .drop('flights_number_7days', 'passengers_number_7days', 'payload_7days', 'av_seat_km_7days')
                       .withColumnRenamed('flights_number_avg', 'flights_number_7days')
                       .withColumnRenamed('passengers_number_avg', 'passengers_number_7days')
                       .withColumnRenamed('payload_avg', 'payload_7days')
                       .withColumnRenamed('av_seat_km_avg', 'av_seat_km_7days'))

# COMMAND ----------

for c in countries_od_weekly.schema.fieldNames():
  countries_od_weekly.schema[c].nullable = True

cols = ['passenger', 'type', 'week_of_year',
        'origin_country', 'origin_iso_country', 'origin_region', 'origin_iso3',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3']

lag_fn = generate_lag_fn(countries_od_weekly.schema, cols)
countries_od_weekly_yoy = countries_od_weekly.groupby('origin_country', 'destination_country', 'passenger').apply(lag_fn)

# COMMAND ----------

countries_od_weekly_wow = (countries_od_weekly_yoy
                           .withColumn('flights_number_7days_wow_change', 
                                       (col('flights_number_7days') - col('lastwk_flights_number_7days'))/col('lastwk_flights_number_7days'))
                           .withColumn('passengers_number_7days_wow_change', 
                                       (col('passengers_number_7days') - col('lastwk_passengers_number_7days'))/col('lastwk_passengers_number_7days'))
                           .withColumn('payload_7days_wow_change',
                                       (col('payload_7days') - col('lastwk_payload_7days'))/col('lastwk_payload_7days'))
                           .withColumn('av_seat_km_7days_wow_change',
                                       (col('av_seat_km_7days') - col('lastwk_av_seat_km_7days'))/col('lastwk_av_seat_km_7days'))
                           .withColumn('flights_number_7days_yoy_change', 
                                       (col('flights_number_7days') - col('lastyr_flights_number_7days'))/col('lastyr_flights_number_7days'))
                           .withColumn('passengers_number_7days_yoy_change',
                                       (col('passengers_number_7days') - col('lastyr_passengers_number_7days'))/col('lastyr_passengers_number_7days'))
                           .withColumn('payload_7days_yoy_change',
                                       (col('payload_7days') - col('lastyr_payload_7days'))/col('lastyr_payload_7days'))
                           .withColumn('av_seat_km_7days_yoy_change',
                                       (col('av_seat_km_7days') - col('lastyr_av_seat_km_7days'))/col('lastyr_av_seat_km_7days')))

# COMMAND ----------

#countries_od_weekly_wow.filter("week_date >= '20191201'").write.mode("overwrite").saveAsTable("aviation.countries_od_weekly")
(countries_od_weekly_wow
 .filter(col('week_date') >= start_date)
 .toPandas().to_csv(out_path + 'countries_od_weekly.csv', index=False))

countries_od_wow.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Country-level aggregations

# COMMAND ----------

countries = (od_flows_5
             .groupby('destination_country', 'type', 'passenger', 'date')
             .agg(fn.sum(col('flights_number')).alias('flights_number'),
                  fn.sum(col('passengers_number')).alias('passengers_number'),
                  fn.sum(col('payload')).alias('payload'),
                  fn.sum(col('av_seat_km')).alias('av_seat_km'),
                  fn.first(col('destination_iso_country')).alias('destination_iso_country'),
                  fn.first(col('destination_region')).alias('destination_region'),
                  fn.first(col('destination_iso3')).alias('destination_iso3')))

# COMMAND ----------

cols = ['passenger', 'type',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3']
lag_fn = generate_lag_fn(countries.schema, cols)

for c in countries.schema.fieldNames():
  countries.schema[c].nullable = True
  
countries_wow = countries.groupby('destination_country', 'passenger', 'type',).apply(lag_fn)

countries_wow.cache()
countries_wow.count()

# COMMAND ----------

#countries_wow.filter("date >= '20191201'").write.mode("overwrite").saveAsTable("aviation.countries")
(countries_wow
 .filter(col('date') >= start_date)
 .toPandas().to_csv(out_path + 'countries.csv', index=False))

# COMMAND ----------

regions = (countries_wow
           .groupby('destination_region', 'type', 'passenger', 'date')
           .agg(fn.sum(col('flights_number')).alias('flights_number'),
                fn.sum(col('passengers_number')).alias('passengers_number'),
                fn.sum(col('payload')).alias('payload'),
                fn.sum(col('av_seat_km')).alias('av_seat_km'))
           .withColumn('destination_iso_country', lit(None).cast(StringType()))
           .withColumn('destination_iso3', lit(None).cast(StringType()))
           .withColumnRenamed('destination_region', 'destination_country')
           .withColumn('destination_region', lit(None).cast(StringType())))

# COMMAND ----------

cols = ['passenger', 'type',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3']
lag_fn = generate_lag_fn(regions.schema, cols)

for c in regions.schema.fieldNames():
  regions.schema[c].nullable = True
  
regions_wow = regions.groupby('destination_country', 'passenger', 'type',).apply(lag_fn)

# COMMAND ----------

countries_regions = countries_wow.unionByName(regions_wow)

# COMMAND ----------

(countries_regions
 .filter((col('date') >= start_date) & (col('date') < current_date))
 .toPandas().to_csv(out_path + 'countries_regions.csv', index=False))

# COMMAND ----------

countries_wow.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Airline-level OD weekly aggregations

# COMMAND ----------

airlines_od = (flights_landed_2
               .withColumn('airline_code', col('operator'))
               .withColumn('day_of_year', fn.dayofyear(col('date')))
               .withColumn('year', fn.year(col('date')))
               .withColumn('day_of_years', 365*col('year')+col('day_of_year')+fn.floor((col('year')-1)/4))
               .withColumn('week_of_year', ((col('day_of_years')+4)/7).cast('int'))
               .groupBy('week_of_year', 'airline_code', 'passenger', 'origin_iata', 'destination_iata')
               .agg(fn.count(lit(0)).alias('flights_number_7days'),
                    fn.sum('n').alias('passengers_number_7days'),
                    fn.sum('payload').alias('payload_7days'))
               .filter("origin_iata is not null")
               .filter("destination_iata is not null"))

# COMMAND ----------

# join with distance dataframe and compute ASK metric
arlines_od_2 = (airlines_od
                .join(df_distance, on=['origin_iata', 'destination_iata'])\
                .withColumn('av_seat_km_7days', col('passengers_number_7days')*col('distance')))

# COMMAND ----------

arlines_od_3 = (arlines_od_2.join(df_codes_2, on=arlines_od_2.origin_iata == df_codes.iata_code, how='inner')
                .withColumnRenamed('iso_country', 'origin_iso_country')
                .withColumnRenamed('iso3', 'origin_iso3')
                .withColumnRenamed('country', 'origin_country')
                .withColumnRenamed('region', 'origin_region')
                .withColumnRenamed('latitude', 'origin_latitude')
                .withColumnRenamed('longitude', 'origin_longitude')
                .drop('iata_code', 'coordinates'))

arlines_od_4 = (arlines_od_3.join(df_codes_2, on=arlines_od_3.destination_iata == df_codes.iata_code, how='inner')
                .withColumnRenamed('iso_country', 'destination_iso_country')
                .withColumnRenamed('iso3', 'destination_iso3')
                .withColumnRenamed('country', 'destination_country')
                .withColumnRenamed('region', 'destination_region')
                .withColumnRenamed('latitude', 'destination_latitude')
                .withColumnRenamed('longitude', 'destination_longitude')
                .drop('iata_code', 'coordinates'))

# COMMAND ----------

arlines_od_5 = (arlines_od_4
                .withColumn('type', (fn.when(col('origin_iso_country') == col('destination_iso_country'), 'domestic')
                                     .otherwise(fn.when(col('origin_region') == col('destination_region'), 'intra-regional')
                                                .otherwise('inter-regional'))))
                .join(week_dates, on=['week_of_year'])
                .withColumn('flights_number_avg', col('flights_number_7days')/col('ndays'))
                .withColumn('passengers_number_avg', col('passengers_number_7days')/col('ndays'))
                .withColumn('payload_avg', col('payload_7days')/col('ndays'))
                .withColumn('av_seat_km_avg', col('av_seat_km_7days')/col('ndays'))
                .drop('flights_number_7days', 'passengers_number_7days', 'payload_7days', 'av_seat_km_7days')
                .withColumnRenamed('flights_number_avg', 'flights_number_7days')
                .withColumnRenamed('passengers_number_avg', 'passengers_number_7days')
                .withColumnRenamed('payload_avg', 'payload_7days')
                .withColumnRenamed('av_seat_km_avg', 'av_seat_km_7days'))

# COMMAND ----------

cols = ['airline_code', 'origin_iata', 'destination_iata', 'passenger', 'type',
        'origin_country', 'origin_iso_country', 'origin_region', 'origin_iso3', 'origin_longitude', 'origin_latitude',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3', 'destination_longitude', 'destination_latitude',
        'week_of_year', 'ndays']
lag_fn = generate_lag_fn(arlines_od_5.schema, cols)

for c in arlines_od_5.schema.fieldNames():
  arlines_od_5.schema[c].nullable = True
  
arlines_od_yoy = arlines_od_5.groupby('airline_code', 'origin_iata', 'destination_iata', 'passenger').apply(lag_fn)

# COMMAND ----------

arlines_od_wow = (arlines_od_yoy
                  .withColumn('flights_number_7days_wow_change', (col('flights_number_7days') - col('lastwk_flights_number_7days'))/col('lastwk_flights_number_7days'))
                  .withColumn('passengers_number_7days_wow_change',
                              (col('passengers_number_7days') - col('lastwk_passengers_number_7days'))/col('lastwk_passengers_number_7days'))
                  .withColumn('payload_7days_wow_change',
                              (col('payload_7days') - col('lastwk_payload_7days'))/col('lastwk_payload_7days'))
                  .withColumn('av_seat_km_7days_wow_change',
                              (col('av_seat_km_7days') - col('lastwk_av_seat_km_7days'))/col('lastwk_av_seat_km_7days'))
                  .withColumn('flights_number_7days_yoy_change', (col('flights_number_7days') - col('lastyr_flights_number_7days'))/col('lastyr_flights_number_7days'))
                  .withColumn('passengers_number_7days_yoy_change',
                              (col('passengers_number_7days') - col('lastyr_passengers_number_7days'))/col('lastyr_passengers_number_7days'))
                  .withColumn('payload_7days_yoy_change',
                              (col('payload_7days') - col('lastyr_payload_7days'))/col('lastyr_payload_7days'))
                  .withColumn('av_seat_km_7days_yoy_change',
                              (col('av_seat_km_7days') - col('lastyr_av_seat_km_7days'))/col('lastyr_av_seat_km_7days')))

# COMMAND ----------

#arlines_od_wow.filter("week_date >= '20191201'").write.mode("overwrite").saveAsTable("aviation.airlines_od")
(arlines_od_wow
 .filter(col('week_date') >= start_date)
 .toPandas().to_csv(out_path + 'airlines_od.csv', index=False))

# COMMAND ----------

arlines_od_wow.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Airline-level weekly aggregations

# COMMAND ----------

airlines = (arlines_od_5
            .groupby('airline_code', 'week_of_year', 'destination_iata', 'type', 'passenger')
            .agg(fn.sum(col('flights_number_7days')).alias('flights_number_7days'),
                 fn.sum(col('passengers_number_7days')).alias('passengers_number_7days'),
                 fn.sum(col('payload_7days')).alias('payload_7days'),
                 fn.sum(col('av_seat_km_7days')).alias('av_seat_km_7days'),
                 fn.first(col('destination_country')).alias('destination_country'),
                 fn.first(col('destination_iso_country')).alias('destination_iso_country'),
                 fn.first(col('destination_region')).alias('destination_region'),
                 fn.first(col('destination_iso3')).alias('destination_iso3'),
                 fn.first(col('week_date')).alias('week_date'),
                 fn.first(col('ndays')).alias('ndays')))

# COMMAND ----------

cols = ['airline_code', 'destination_iata', 'passenger', 'type',
        'destination_country', 'destination_iso_country', 'destination_region', 'destination_iso3',
        'week_of_year', 'ndays']
lag_fn = generate_lag_fn(airlines.schema, cols)

for c in airlines.schema.fieldNames():
  airlines.schema[c].nullable = True
  
airlines_wow = airlines.groupby('airline_code', 'destination_iata', 'passenger', 'type').apply(lag_fn)

# COMMAND ----------

#airlines_wow.filter("week_date >= '20191201'").write.mode("overwrite").saveAsTable("aviation.airlines")
(airlines_wow
 .filter(col('week_date') >= start_date)
 .toPandas().to_csv(out_path + 'airlines.csv', index=False))

# COMMAND ----------

