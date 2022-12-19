package predictor

import org.apache.spark.sql.SQLContext

class Counter(val sqlContext: SQLContext) {

  def count(): Unit ={
        val df2007 = sqlContext.read.option("header",true).csv("data/airflightDelays/2007.csv.bz2")
        println(df2007.schema)
        df2007.createOrReplaceTempView("data")
        println("The number of flight departing in the early morning: ")
        sqlContext.sql("SELECT COUNT(data.FlightNum) FROM data WHERE data.DepTime BETWEEN 0 AND 600").show()
        println("The number of flight departing in the morning: ")
        sqlContext.sql("SELECT COUNT(data.FlightNum) FROM data WHERE data.DepTime BETWEEN 601 AND 1000").show()
        println("The number of flight departing in the noon: ")
        sqlContext.sql("SELECT COUNT(data.FlightNum) FROM data WHERE data.DepTime BETWEEN 1001 AND 1400").show()
        println("The number of flight departing in the afternoon: ")
        sqlContext.sql("SELECT COUNT(data.FlightNum) FROM data WHERE data.DepTime BETWEEN 1401 AND 1900").show()
        println("The number of flight departing in the evening: ")
        sqlContext.sql("SELECT COUNT(data.FlightNum) FROM data WHERE data.DepTime BETWEEN 1901 AND 2359").show()

        //find which destination have most on-time flights
        val queryDestResult = sqlContext.sql("SELECT DISTINCT data.Dest, COUNT(data.ArrDelay) AS delayTimes FROM data where data.ArrDelay = 0 GROUP BY data.Dest ORDER BY delayTimes DESC")
        println(queryDestResult.head(5).mkString("Array(", ", ", ")"))

        //delay count
        val queryOriginResult = sqlContext.sql("SELECT DISTINCT data.Origin, data.DepDelay FROM data where data.DepDelay > 60 ORDER BY data.DepDelay DESC")
        println(queryOriginResult.head(5).mkString("Array(", ", ", ")"))
  }
}
