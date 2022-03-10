from pyspark.sql import *
df = spark \
    .read \
    .format("csv")\
    .option("header", "true") \
    .load("/interview-datasets/sa/airlines/part-00000")

airline_schema = df.schema

df = spark \
    .read \
    .format("csv")\
    .schema(airline_schema) \
    .load("/interview-datasets/sa/airlines/*")

display(df)


#df.agg(count_distinct(df.UniqueCarrier).alias('c')).collect()
unique_airflines=df.groupBy('FlightNum').count()
count_unique_airflines=unique_airflines.count()
print("Unique Airlines: "+str(count_unique_airflines))

delayed_df = df.withColumn('Delay', ( df['DepTime'] - df['CRSDepTime'] ) )
delayed_df = delayed_df.where(delayed_df.Delay>0)
#delayed_df.select("Delay").show()
#display(delayed_df.groupBy('UniqueCarrier').count()).head(5)
delayed_df=delayed_df.groupBy('UniqueCarrier').count()
delayed_df=delayed_df.sort(delayed_df['count'].desc())
display(delayed_df, 5)
#delayed_df.printSchema()
#print(delayed_df.select('UniqueCarrier').where(delayed_df.Delay>0).count())
#print(delayed_df.count())
#print(delayed_df.groupBy('UniqueCarrier').count(), 5)

# Please provide your code answer for Question 4
from pyspark.sql.functions import avg,round
avg_delay=df.groupBy('FlightNum').agg(avg('ArrDelay').alias('Avg_ArrDelay'))
avg_delay=avg_delay.select('FlightNum',round('Avg_ArrDelay',2).alias('Avg_ArrDelay'))
avg_delay.show()

"""PYSPARK UDF CREATION"""
from pyspark.sql.types import StringType
def status(avg_delay):
  status=''
  if avg_delay>15:
    status='Late'
  elif avg_delay<0:
    status='Early'
  elif avg_delay>=0 and avg_delay<=15:
    status='On-time'
  return status

statusUDF=udf(lambda z: status(z),StringType())

from pyspark.sql.functions import col
avg_delay.select("FlightNum","Avg_ArrDelay",statusUDF(col("Avg_ArrDelay")).alias("status") ).show()


