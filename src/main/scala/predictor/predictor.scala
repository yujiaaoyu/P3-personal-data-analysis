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
import org.joda.time.format.DateTimeFormat
import org.joda.time.DateTime
import org.joda.time.Days
import java.io._
import scala.collection.JavaConverters._

//reference: https://medium.com/@pedrodc/building-a-big-data-machine-learning-spark-application-for-flight-delay-prediction-4f9507cdb010
object DelayRecProject {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("DelayRecProject")
    val sc = new SparkContext(conf)

    sc.setLogLevel("WARN")

    val sqlContext = new SQLContext(sc)

    val counter = new Counter(sqlContext)
    counter.count()

    //delay-predict
    //step1: pre-process date
    val tmp2007Rdd = prepFlightDelays("data/airflightDelays/2007.csv.bz2",sc)
    val mapped2007Rdd = tmp2007Rdd.map(rec => rec.gen_features._2)
    //for testing
    //mapped2007Rdd.take(5).map(x => x mkString ",").foreach(println)
    val mapped2008Rdd = prepFlightDelays("data/airflightDelays/2008.csv.bz2",sc).map(rec => rec.gen_features._2)

    //step2: use Spark and ML-Lib train models
    // Prepare training set
    val parsedTrainData = mapped2007Rdd.map(parseData)
    parsedTrainData.cache
    val myScaler = new StandardScaler(withMean = true, withStd = true).fit(parsedTrainData.map(x => x.features))
    //    //convert the featuredRDD containing the features array to a new RDD containing the labeled points
    val myTrainData = parsedTrainData.map(x => LabeledPoint(x.label, myScaler.transform(Vectors.dense(x.features.toArray))))
    myTrainData.cache

    // Prepare test/validation set
    val parsedTestData = mapped2008Rdd.map(parseData)
    parsedTestData.cache
    val myTestData = parsedTestData.map(x => LabeledPoint(x.label, myScaler.transform(Vectors.dense(x.features.toArray))))
    myTestData.cache

    //for testing
    //myTrainData.take(3).map(x => (x.label, x.features)).foreach(println)

    //step3: evaluation metrics
    // Function to compute evaluation metrics
    def evaluate_metrics(labelsAndPreds: RDD[(Double, Double)]) : Tuple2[Array[Double], Array[Double]] = {
      val tp = labelsAndPreds.filter(r => r._1==1 && r._2==1).count.toDouble
      val tn = labelsAndPreds.filter(r => r._1==0 && r._2==0).count.toDouble
      val fp = labelsAndPreds.filter(r => r._1==1 && r._2==0).count.toDouble
      val fn = labelsAndPreds.filter(r => r._1==0 && r._2==1).count.toDouble

      val precision = tp / (tp+fp)
      val recall = tp / (tp+fn)
      val F_measure = 2*precision*recall / (precision+recall)
      val accuracy = (tp+tn) / (tp+tn+fp+fn)
      Tuple2(Array(tp, tn, fp, fn), Array(precision, recall, F_measure, accuracy))
    }

    class MyMetrics(labelsAndPreds: RDD[(Double, Double)]) extends java.io.Serializable {

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
      .run(myTrainData)
    // Predict
    val labelsAndPreds_lr = myTestData.map { point =>
      val pred = model_lr.predict(point.features)
      (pred, point.label)
    }
    val m_lr = evaluate_metrics(labelsAndPreds_lr)._2
    println("Logistic Regression - precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_lr(0), m_lr(1), m_lr(2), m_lr(3)))

    //step5: Build the SVM model
    val svmAlgo = new SVMWithSGD()
    svmAlgo.optimizer.setNumIterations(100)
      .setRegParam(1.0)
      .setStepSize(1.0)
    val model_svm = svmAlgo.run(myTrainData)

    //Predict
    val labelAndPredict_svm = myTestData.map { point =>
      val pred = model_svm.predict(point.features)
      (pred, point.label)
    }
    val m_svm = evaluate_metrics(labelAndPredict_svm)._2
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
    val m_dt = evaluate_metrics(labelsAndPreds_dt)._2
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
    val m_rf = new MyMetrics(labelsAndPreds_rf)
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

    def gen_features: (String, Array[Double]) = {
      val values = Array(
        depDelay.toDouble,
        month.toDouble,
        dayOfMonth.toDouble,
        dayOfWeek.toDouble,
        get_hour(crsDepTime).toDouble,
        distance.toDouble,
      )
      new Tuple2(to_date(year.toInt, month.toInt, dayOfMonth.toInt), values)
    }

    def get_hour(depTime: String) : String = "%04d".format(depTime.toInt).take(2)
    def to_date(year: Int, month: Int, day: Int) = "%04d%02d%02d".format(year, month, day)

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