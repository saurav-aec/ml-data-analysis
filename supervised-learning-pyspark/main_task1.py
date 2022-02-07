from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext
#from pyspark.ml.clustering import GaussianMixture
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# battery_power,0
# blue,1
# clock_speed,2
# dual_sim,3
# fc,4
# four_g,5
# int_memory,6
# m_dep,7
# mobile_wt,8
# n_cores,9
# pc,10
# px_height,11
# px_width,12
# ram,13
# sc_h,14
# sc_w,15
# talk_time,16
# three_g,17
# touch_screen,18
# wifi,19
# price_range 20

def parsePoint(record):
    return LabeledPoint(1 if int(record[0]) >= 1 else 0, record[1:])

def cleanData(record):
    if len(record) != 11:
        return False
    elif None in record:
        return False
    else:
        return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: MobilePricePredictorLR <trainFile> <testFile> ", 
        file=sys.stderr)
        exit(-1)
        
    trainingFile = sys.argv[1]
    testingFile = sys.argv[2]

    sc = SparkContext(appName="MobilePricePredictorLR")
    mobileRdd = sc.textFile(trainingFile)
    testMobileRdd = sc.textFile(testingFile)


    featureRdd = mobileRdd.map(
        lambda r: r.split(',')).map(
            lambda col: array([int(col[20]), float(col[0]), float(col[2]), float(col[4]),
             float(col[6]), float(col[7]), float(col[8]), float(col[9]),
              float(col[10]), float(col[11]), float(col[12])])).filter(
                  lambda x: cleanData(x)).map(parsePoint)

    testFeatureRdd = testMobileRdd.map(
        lambda r: r.split(',')).map(
            lambda col: array([int(col[20]), float(col[0]), float(col[2]), float(col[4]),
             float(col[6]), float(col[7]), float(col[8]), float(col[9]),
              float(col[10]), float(col[11]), float(col[12])])).filter(
                  lambda x: cleanData(x)).map(parsePoint)


    model = LogisticRegressionWithLBFGS.train(featureRdd,
     iterations= 10000, regParam= 0.5)

    labelsAndPreds = testFeatureRdd.map(
        lambda p: (p.label, model.predict(p.features)))

    tp = labelsAndPreds.filter(
        lambda r: (r[0] == 1) and (r[0] == r[1])).count()
    tn = labelsAndPreds.filter(
        lambda r: (r[0] == 0) and (r[0] == r[1])).count()
    fp = labelsAndPreds.filter(
        lambda r: (r[0] == 0) and (r[0] != r[1])).count()
    fn = labelsAndPreds.filter(
        lambda r: (r[0] == 1) and (r[0] != r[1])).count()

    precision = 0.
    recall = 0.

    if tp == 0 and fp == 0:
        precision = 0
    else:
        precision = tp/(tp + fp)

    if tp == 0 and fn == 0:
        recall = 0
    else:
        recall = tp/(tp + fn)

    f1 = 0.
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*(recall * precision) / (recall + precision)

    print("F1 = ", f1, ", Precision = ", precision, ", Recall = ", recall)

    
    sc.stop()
