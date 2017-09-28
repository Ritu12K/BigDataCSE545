
# coding: utf-8

# In[1]:

import sys
import string
from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Manager
from pprint import pprint
import numpy as np
from random import random
from pyspark import SparkConf, SparkContext
from operator import add


def main():

    conf = SparkConf().setAppName("BDA").setMaster("local")
    sc = SparkContext(conf=conf)

    data = [(1, "The horse raced past the barn fell"),
            (2, "The complex houses married and single soldiers and their families"),
            (3, "There is nothing either good or bad, but thinking makes it so"),
            (4, "I burn, I pine, I perish"),
            (5, "Come what come may, time and the hour runs through the roughest day"),
            (6, "Be a yardstick of quality."),
            (7, "A horse is the projection of peoples' dreams about themselves - strong, powerful, beautiful"),
            (8, "I believe that at the end of the century the use of words and general educated opinion will have altered so much that one will be able to speak of machines thinking without expecting to be contradicted."),
            (9, "The car raced past the finish line just in time."),
            (10, "Car engines purred and the tires burned.")]

    data1 = [('R', ['apple', 'orange', 'pear', 'blueberry']),
	 		 ('S', ['pear', 'orange', 'strawberry', 'fig', 'tangerine', 'Apple'])]
    data2 = [('R', [x for x in range(50) if random() > 0.5]),
	 		 ('S', [x for x in range(50) if random() > 0.75])]
    rdd = sc.parallelize(data)

#Word count implementation below
    count_word=rdd.flatMap(lambda line: line[1].split(' ')).map(lambda w: w.lower()).map(lambda word:(word,1)).reduceByKey(lambda v1,v2: v1+v2)

    print(count_word.sortByKey().collect())

#Set Difference Implementation below:
    set_rdd=sc.parallelize(data1)

    set_diff= set_rdd.flatMap(lambda kv: [(value,kv[0])for value in kv[1]]).reduceByKey(lambda v1,v2: (v1,v2)).filter(lambda x: 'R' in x[1] and 'S' not in x[1]).map(lambda x: x[0])

    print(set_diff.collect())


if __name__=="__main__":
    main()



