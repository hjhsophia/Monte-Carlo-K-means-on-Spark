import sys
import re
import os
from pyspark import SparkContext

if __name__=="__main__":
#	if len(sys.argv)<2:
#		print >> sys.stderr, "Usage: step1 <file>"
#		exit(-1)

	sc=SparkContext()
        DIR=os.getcwd()

	dt=sc.textFile(sys.argv[1])\
	.map(lambda x:x.encode("ascii","ignore")).map(lambda x:re.split(",|/|\|",x)).filter(lambda x:len(x)==7)\
	.map(lambda x: [float(x[5]),float(x[6])]).map(lambda x:re.sub('\[|,|\]|\'','',str(x)))\
	.saveAsTextFile("/wordcities")


	sc.stop()

 

