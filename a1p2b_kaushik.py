
# coding: utf-8

# In[1]:

import re
import sys
import string
from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Manager
from pprint import pprint
import numpy as np
from random import random
from pyspark import SparkConf, SparkContext
from operator import add


def removePunctuation(x):
    for char in string.punctuation:
        s = x[1].replace(char, " ")
    return x[0], s


def main():
    conf = SparkConf().setAppName("BDA").setMaster("local")
    sc = SparkContext(conf=conf)


# Create an rdd of all the filenames 
# (your code should have a variable defined with the directory where all the files are stored)
    path_to_directory="file:///Users/ritukaushik/Desktop/blogs/5*.xml"
    directory_files = sc.wholeTextFiles(path_to_directory)


# Use transformations until you are left with only a set of possible industries
    file_names=directory_files.keys()

    files=file_names.map(lambda string: string.split("/")[5])

    industries=files.map(lambda partial_filenames: partial_filenames.split(".")[3])

    distinct_industries=industries.distinct()

# Use an action to export the rdd to a set and make this a spark broadcast variable
    industry_broadcast=sc.broadcast(distinct_industries.collect()) 

# Create an rdd for the contents of all files [i.e. sc.wholeTextFiles(file1,file2,...) ]
    files_content=directory_files.values()

    posts_by_dates=files_content.map(lambda single_file_content: single_file_content.split('<date>'))

    posts=posts_by_dates.flatMap(lambda x: x[1:])

    new_post=posts.map(lambda x: x.split('</date>'))

    new = new_post.map(lambda x: removePunctuation(x))

    new = new.map(lambda x: ([x[0],x[1]]))

    test_this=new.flatMap(lambda x: [(word, x[0]) for word in x[1].split(' ')])

    result=test_this.filter(lambda x: x[0] in industry_broadcast.value)

    answer=result.map(lambda x: (x[0], x[1].split(',')[1:]))

    output=answer.map(lambda x: (x[0],(x[1][1],x[1][0])) )

    output=answer.map(lambda x: (x[0],str(x[1][1] +"-" + x[1][0])) )

    output=output.map(lambda x: (x[0], (x[1],1)))

    output_new = output.map(lambda x: [(x[0],x[1][0]),x[1][1]])

    final_output=output_new.reduceByKey(lambda x,y: x+y)

    final_result=final_output.map(lambda x: (x[0][0],(x[0][1],x[1])))

    res=final_result.groupByKey()

    print(res.mapValues(list).collect())

if __name__=="__main__":
    main()



# In[ ]:



