%scala

import java.net.URL
import java.io.File
import org.apache.commons.io.FileUtils

val tmpFile = new File("/tmp/rows.json")
FileUtils.copyURLToFile(new URL("https://health.data.ny.gov/api/views/jxy9-yhdk/rows.json?accessType=DOWNLOAD"), tmpFile)

dbutils.fs.cp('file:/tmp/rows.json', userhome + '/rows.json')
dbutils.fs.cp(userhome + '/rows.json' ,f"dbfs:/tmp/{user}/rows.json")
baby_names_path = f"dbfs:/tmp/{user}/rows.json"

dbutils.fs.head(baby_names_path)
# Ensure you use baby_names_path to answer the questions. A bug in Spark 2.X will cause your read to fail if you read the file from userhome. 

from pyspark.sql.functions import to_json,col
import re
import pandas as pd
df = spark.read.option("multiline","true").json(baby_names_path)
df_json=df.select(to_json(col("data")).alias("json_data"))
data_list=df_json.select('json_data').collect()
newlist=data_list[0].json_data
updated_list=re.sub('null', '"null"', newlist)
updated_list=eval(updated_list)
df_pd_json=pd.DataFrame(updated_list,columns=["sid", "id", "position", "created_at", "created_meta", "updated_at", "updated_meta", "meta", "year", "first_name", "county", "sex", "count"])
json_df = spark.createDataFrame(df_pd_json)
display(json_df)

json_df.createOrReplaceTempView("baby_names")

# Please provide your code answer for Question 2 here. You will need separate cells for your SQL answer and your Python or Scala answer.
"""Using SQL"""
spark.sql("""
  select year,first_name
  from
  (select year,first_name,cnt,DENSE_RANK () OVER ( 
			PARTITION BY year
			ORDER BY cnt DESC
		) as rnk
  from
  (
  select year,first_name,count(1) as cnt
  from baby_names
  group by year,first_name
    ) tmp ) tmp1
  where rnk=1
  order by year
""").show()

"""Using PySpark"""
from pyspark.sql.window import Window
from pyspark.sql.functions import dense_rank
windowSpec = Window.partitionBy("year").orderBy(col("count").desc())
baby_names=json_df.groupBy(['year','first_name']).count()
baby_names=baby_names.withColumn("rnk",dense_rank().over(windowSpec))
baby_names.where(col("rnk")==1).orderBy(col("year")).select("year","first_name").show()

visitors_path = "/interview-datasets/sa/births/births-with-visitor-data.json"


## Hint: the code below will read in the downloaded JSON files. However, the xml column needs to be given structure. Consider using a UDF.
df_xml = spark.read.option("inferSchema", True).json(visitors_path)
display(df_xml)

#df_xml.groupBy('county').count().show()
df_visitors=df_xml.select('county','visitors')
df2 = df_visitors.selectExpr('county',
    "xpath(visitors, './visitors/visitor/@id') id",
    "xpath(visitors, './visitors/visitor/@age') age",
    "xpath(visitors, './visitors/visitor/@sex') sex"
).selectExpr("county",
    "explode(arrays_zip(id, age, sex)) visitors"
).select('county','visitors.*')

df2.show(truncate=False)

#df_xml.groupBy('county').count().show()
df_visitors_new=df_xml.select('sid','county','visitors')
df3 = df_visitors_new.selectExpr('sid','county',
    "xpath(visitors, './visitors/visitor/@id') id",
    "xpath(visitors, './visitors/visitor/@age') age",
    "xpath(visitors, './visitors/visitor/@sex') sex"
).selectExpr("sid","county",
    "explode(arrays_zip(id, age, sex)) visitors"
).select('sid','county','visitors.*')

df3.show(truncate=False)

before=df_xml.count()
after=df2.count()
print("Before: {}\nAfter: {}".format(before,after))


## Hint: check for inconsistently capitalized field values. It will make your answer incorrect.
from pyspark.sql.functions import avg
avg_vis=df2.groupBy('county','id').count().select('county','id')
avg_vis=avg_vis.groupBy('county').agg(avg('id').alias('avg_visitors'))
windowSpec = Window.orderBy(col("avg_visitors").desc())
avg_vis=avg_vis.withColumn("rnk",dense_rank().over(windowSpec))
avg_vis.where(col("rnk")==1).select("county").show()

## Hint: check for inconsistently capitalized field values. It will make your answer incorrect.
from pyspark.sql.functions import round
avg_age=df3.where('county=="KINGS"')
avg_age.groupBy('sid').agg(round(avg('age'),1).alias('avg_age')).orderBy(col('avg_age').desc()).show()

## Hint: check for inconsistently capitalized field values. It will make your answer incorrect.
com_vis=df3.where(col('county')=="KINGS")
#com_vis.groupBy('sid','id').count().orderBy(col('count').desc()).limit(1).show()
com_vis_new=com_vis.groupBy('id').count().orderBy(col('count').desc())
windowSpec = Window.orderBy(col("count").desc())
com_vis_new=com_vis_new.withColumn("rnk",dense_rank().over(windowSpec))
com_vis_new=com_vis_new.where(col("rnk")==1).select("id")
com_vis_new.show()



mvv_list = com_vis_new.selectExpr("id as mvv")
mvv_count = [int(row.mvv) for row in mvv_list.collect()]

com_vis.filter(col('id').isin(mvv_count)).select('id','age').show()