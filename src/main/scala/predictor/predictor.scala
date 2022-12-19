package predictor

import au.com.bytecode.opencsv.CSVReader
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.rdd._
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import java.io._
import scala.collection.JavaConverters._


object DelayRecProject {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("DelayRecProject")
    val sc = new SparkContext(conf)

    sc.setLogLevel("WARN")

    val sqlContext = new SQLContext(sc)

    val counter = new Counter(sqlContext)
    counter.count()

    //delay-predict
    //reference: https://medium.com/@pedrodc/building-a-big-data-machine-learning-spark-application-for-flight-delay-prediction-4f9507cdb010
    //step1: pre-process date
    val data_2007tmp = prepFlightDelays("data/airflightDelays/2007.csv.bz2",sc)
    val data_2007 = data_2007tmp.map(rec => rec.gen_features._2)


        val data_2008 = prepFlightDelays("data/airflightDelays/2008.csv.bz2",sc).map(rec => rec.gen_features._2)
        data_2007.take(5).map(x => x mkString ",").foreach(println)

        //step2: use Spark and ML-Lib train models
        // Prepare training set
        val parsedTrainData = data_2007.map(parseData)
        parsedTrainData.cache
        val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedTrainData.map(x => x.features))
        val scaledTrainData = parsedTrainData.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
        scaledTrainData.cache

        // Prepare test/validation set
        val parsedTestData = data_2008.map(parseData)
        parsedTestData.cache
        val scaledTestData = parsedTestData.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
        scaledTestData.cache

        scaledTrainData.take(3).map(x => (x.label, x.features)).foreach(println)

        //step3: evaluation metrics
        // Function to compute evaluation metrics
        def eval_metrics(labelsAndPreds: RDD[(Double, Double)]) : Tuple2[Array[Double], Array[Double]] = {
          val tp = labelsAndPreds.filter(r => r._1==1 && r._2==1).count.toDouble
          val tn = labelsAndPreds.filter(r => r._1==0 && r._2==0).count.toDouble
          val fp = labelsAndPreds.filter(r => r._1==1 && r._2==0).count.toDouble
          val fn = labelsAndPreds.filter(r => r._1==0 && r._2==1).count.toDouble

          val precision = tp / (tp+fp)
          val recall = tp / (tp+fn)
          val F_measure = 2*precision*recall / (precision+recall)
          val accuracy = (tp+tn) / (tp+tn+fp+fn)
          new Tuple2(Array(tp, tn, fp, fn), Array(precision, recall, F_measure, accuracy))
        }

        class Metrics(labelsAndPreds: RDD[(Double, Double)]) extends java.io.Serializable {

          private def filterCount(lftBnd:Int,rtBnd:Int):Double = labelsAndPreds
            .map(x => (x._1.toInt, x._2.toInt))
            .filter(_ == (lftBnd,rtBnd)).count()

          lazy val tp = filterCount(1,1)  // true positives
          lazy val tn = filterCount(0,0)  // true negatives
          lazy val fp = filterCount(0,1)  // false positives
          lazy val fn = filterCount(1,0)  // false negatives

          lazy val precision = tp / (tp+fp)
          lazy val recall = tp / (tp+fn)
          lazy val F1 = 2*precision*recall / (precision+recall)
          lazy val accuracy = (tp+tn) / (tp+tn+fp+fn)
        }

        //step4: Build the Logistic Regression model
        val model_lr = new LogisticRegressionWithLBFGS()
          .setNumClasses(2)
          .setIntercept(true)
          .run(scaledTrainData)
        // Predict
        val labelsAndPreds_lr = scaledTestData.map { point =>
          val pred = model_lr.predict(point.features)
          (pred, point.label)
        }
        val m_lr = eval_metrics(labelsAndPreds_lr)._2
        println("Logistic Regression - precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_lr(0), m_lr(1), m_lr(2), m_lr(3)))

        //step5: Build the SVM model
        val svmAlg = new SVMWithSGD()
        svmAlg.optimizer.setNumIterations(100)
          .setRegParam(1.0)
          .setStepSize(1.0)
        val model_svm = svmAlg.run(scaledTrainData)

        //Predict
        val labelsAndPreds_svm = scaledTestData.map { point =>
          val pred = model_svm.predict(point.features)
          (pred, point.label)
        }
        val m_svm = eval_metrics(labelsAndPreds_svm)._2
        println("SVM model precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_svm(0), m_svm(1), m_svm(2), m_svm(3)))

        // step6: Build the Decision Tree model
        val numClasses = 2
        val categoricalFeaturesInfo = Map[Int, Int]()
        val impurity = "gini"
        val maxDepth = 10
        val maxBins = 100
        val model_dt = DecisionTree.trainClassifier(parsedTrainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

        // Predict
        val labelsAndPreds_dt = parsedTestData.map { point =>
          val pred = model_dt.predict(point.features)
          (pred, point.label)
        }
        val m_dt = eval_metrics(labelsAndPreds_dt)._2
        println("Decision Tree precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_dt(0), m_dt(1), m_dt(2), m_dt(3)))

        //step7: build random forest model
        val treeStrategy = Strategy.defaultStrategy("Classification")
        val numTrees = 100
        val featureSubsetStrategy = "auto" // Let the algorithm choose
        val model_rf = RandomForest.trainClassifier(parsedTrainData, treeStrategy, numTrees, featureSubsetStrategy, seed = 123)

        // Predict
        val labelsAndPreds_rf = parsedTestData.map { point =>
          val pred = model_rf.predict(point.features)
          (point.label, pred)
        }
        val m_rf = new Metrics(labelsAndPreds_rf)
        println("Random forest precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f"
          .format(m_rf.precision, m_rf.recall, m_rf.F1, m_rf.accuracy))

      }

      case class DelayRec(year: String,
                          month: String,
                          dayOfMonth: String,
                          dayOfWeek: String,
                          crsDepTime: String,
                          depDelay: String,
                          origin: String,
                          distance: String,
                          cancelled: String) {

        val holidays = List("01/01/2007", "01/15/2007", "02/19/2007", "05/28/2007", "06/07/2007", "07/04/2007",
          "09/03/2007", "10/08/2007" ,"11/11/2007", "11/22/2007", "12/25/2007",
          "01/01/2008", "01/21/2008", "02/18/2008", "05/22/2008", "05/26/2008", "07/04/2008",
          "09/01/2008", "10/13/2008" ,"11/11/2008", "11/27/2008", "12/25/2008")

        def gen_features: (String, Array[Double]) = {
          val values = Array(
            depDelay.toDouble,
            month.toDouble,
            dayOfMonth.toDouble,
            dayOfWeek.toDouble,
            get_hour(crsDepTime).toDouble,
            distance.toDouble,
            days_from_nearest_holiday(year.toInt, month.toInt, dayOfMonth.toInt)
          )
          new Tuple2(to_date(year.toInt, month.toInt, dayOfMonth.toInt), values)
        }

        def get_hour(depTime: String) : String = "%04d".format(depTime.toInt).take(2)
        def to_date(year: Int, month: Int, day: Int) = "%04d%02d%02d".format(year, month, day)

        def days_from_nearest_holiday(year:Int, month:Int, day:Int): Int = {
          val sampleDate = new DateTime(year, month, day, 0, 0)

          holidays.foldLeft(3000) { (r, c) =>
            val holiday = DateTimeFormat.forPattern("MM/dd/yyyy").parseDateTime(c)
            val distance = Math.abs(Days.daysBetween(holiday, sampleDate).getDays)
            math.min(r, distance)
          }
        }
      }

      def prepFlightDelays(infile: String, sc: SparkContext): RDD[DelayRec] = {

        val data = sc.textFile(infile)

        data.map { line =>
          val reader = new CSVReader(new StringReader(line))
          reader.readAll().asScala.toList.map(rec => DelayRec(rec(0),rec(1),rec(2),rec(3),rec(5),rec(15),rec(16),rec(18),rec(21)))
        }.map(list => list.head)
          .filter(rec => rec.year != "Year")
          .filter(rec => rec.cancelled == "0")
          .filter(rec => rec.origin == "ORD")
      }

      def parseData(vals: Array[Double]): LabeledPoint = {
        LabeledPoint(if (vals(0)>=15) 1.0 else 0.0, Vectors.dense(vals.drop(1)))
      }
}