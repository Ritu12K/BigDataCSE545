import io
import sys
import numpy as np
import scipy.stats as ss
import zipfile
import string
import hashlib
from PIL import Image 
from numpy.linalg import svd
from pyspark import SparkConf, SparkContext
from tifffile import TiffFile
from scipy import linalg 
from scipy.spatial import distance

def getOrthoTif(zfBytes):
    bytesio = io.BytesIO(zfBytes)
    zfiles = zipfile.ZipFile(bytesio, "r")
    for fn in zfiles.namelist():
        if fn[-4:] == '.tif':
            tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
            return tif.asarray()


def divide_into_blocks(x,t):
    down =  range(0,x.shape[0],t)
    across = range(0,x.shape[1],t)
    reshaped = []
    for d in down:
        for a in across:
            reshaped.append(x[d:d+t,a:a+t,:])
    return reshaped

def pull_2dimage(in_array):
    rows = in_array.shape[0]    
    cols = in_array.shape[1]    
    out_array=[]
    for r in range(rows):
        out_row=[]
        for c in range(cols): 
            nsum=int(in_array[r,c,0])+int(in_array[r,c,1])+int(in_array[r,c,2])
            nmean=nsum/3
            nintensity= int(nmean*(int(in_array[r,c,3])/100))
            out_row.append(int(nintensity))
        out_array.append(out_row)
    return np.array(out_array)

def small_factor(in_array, factor):
    down=range(0,in_array.shape[0],factor)
    across=range(0,in_array.shape[1],factor)
    mean_array=[]
    for d in down:
        this_row=[]
        for a in across:
            m=in_array[d:d+factor,a:a+factor]
            t=np.array(m)
            mn=np.mean(t)
            this_row.append(mn)
        mean_array.append(this_row)
    return np.array(mean_array)

def find_diff(in_array):
    row_diff=np.diff(in_array, axis=1) 
    col_diff=np.diff(in_array, axis=0) 
    flat_array=np.append(row_diff, col_diff)

    for i in range(len(flat_array)) :
        if flat_array[i]>1:
            flat_array[i]=1
        elif flat_array[i]<-1:
            flat_array[i]=-1
        else:
            flat_array[i]=0
    return flat_array


def make_128(in_array):
    feature1 = range(0,3496,38)
    feature2= range(3496,4900,39)
    signature = []
    for chunk in feature1:
        hashcode=hashlib.md5((in_array[chunk:chunk+38])).hexdigest()
        b=bin(int(hashcode,16))
        int_n=int(b, 2)
        signature.append(int_n & 1)
    for chunk1 in feature2:
        hashlib.md5((in_array[chunk1:chunk1+38])).hexdigest()
        b=bin(int(hashcode,16))
        int_n=int(b, 2)
        signature.append(int_n & 1)
    return signature


def divide_into_bands(in_array, no_of_bands):
    bands=[]
    chunk_s=int(128/no_of_bands)
    signature= range(0,128,chunk_s)
    for i in signature:
        bands.append(in_array[i:i+chunk_s])
    return np.array(bands)


def hash_to_buckets(in_array, i, no_of_buckets):
    hashed_code=hashlib.md5(in_array).hexdigest()
    t=int(hashed_code,16)
    return (t%no_of_buckets)+(no_of_buckets*i)

def get_svdPlanes(list_of_images):
    image_matrix=[]
    for sub_image in list_of_images:
        image_matrix.append(sub_image[1])
        
    mu=np.mean(image_matrix, axis=0)
    std=np.std(image_matrix, axis=0)
    img_diffs_zs = (image_matrix - mu) / std
    U, s, V = linalg.svd(img_diffs_zs, full_matrices=1)
    Vnew=V[0:10,:]
    return np.transpose(Vnew)

def reduce_dimensions(feature, evectors):
    return np.matmul(feature, evectors)


def eu_dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))


def main():
    
    sc = SparkContext()

    data_path = 'hdfs:/data/large_sample'
    rdd = sc.binaryFiles(data_path)

    answer=rdd.map(lambda x: ( x[0].split("/")[-1:], getOrthoTif(x[1])))

    res=answer.flatMap(lambda x:[(x[0][0] +"-" +str(i),p) for i, p in enumerate(divide_into_blocks(x[1],500))])

    q=res.filter(lambda x: x[0] in ('3677454_2025195.zip-0','3677454_2025195.zip-1','3677454_2025195.zip-18','3677454_2025195.zip-19')).collect()
    print('\n')
    print("output for 1(e).................")
    print('\n')

    for i in range(len(q)):
        print(q[i][0],q[i][1][0][0])

    print('\n')

    intensity_list=res.map(lambda x: (x[0],pull_2dimage(x[1]))).persist()

    factor=10
    reduced_images=intensity_list.map(lambda x:(x[0],small_factor(x[1],factor)))
    
    r=reduced_images.map(lambda x: (x[0], find_diff(x[1])))

    v=r.filter(lambda x: x[0] in ('3677454_2025195.zip-1', '3677454_2025195.zip-18')).collect()

    print('\n')
    print("output for 2(f) features reduced to size 10.................")
    print('\n')

    for i in range(len(v)):
        print(v[i][0],v[i][1])

    print('\n')

    signatures=r.map(lambda x: (x[0], make_128(x[1])))

    no_of_bands=2
    no_of_buckets=700
    bands=signatures.map(lambda x: (x[0], divide_into_bands(x[1], no_of_bands)))

    hashed_bands=bands.flatMap(lambda x:[(x[0], hash_to_buckets(x[1][i], i,no_of_buckets)) for i in range(0,no_of_bands)])

    group_by_buckets=hashed_bands.map(lambda x: (x[1], x[0])).reduceByKey(lambda x, y: np.append(x,y))

    filter_images=group_by_buckets.filter(lambda x: any(item in ('3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19') for item in x[1]))

    z=filter_images.flatMap(lambda x: [(x[1][i], x[1]) for i in range(len(x[1]))])

    needed_neigbours=z.filter(lambda x: x[0] in ('3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19'))

    y=needed_neigbours.reduceByKey(lambda x, y: np.append(x,y))

    s=y.map(lambda x: (x[0], list(set(x[1]))))
    sample_size=30
    rddsample=r.takeSample(False,sample_size)

    evectors=get_svdPlanes(rddsample)

    similar_images=s.filter(lambda x: x[0] in ('3677454_2025195.zip-1', '3677454_2025195.zip-18')).collect()
    
    print('\n')
    print("output for 3(b): potential candidates found for each")
    print('\n')
    for i in range(len(similar_images)):
        print(similar_images[i])
        print('\n')
                   
    
    print('\n')

    l=[]
    for i in range(len(similar_images)):
        for j in range (len(similar_images[i][1])):
            l.append(similar_images[i][1][j])

    neighbours=list(set(l))

    n_features=r.filter(lambda x: x[0] in neighbours).map(lambda x: (x[0], reduce_dimensions(x[1], evectors))).collectAsMap()

    dist_matrix=[]
    for i in range (len(similar_images)):
        pair_dist=[]
        for j in range (len(similar_images[i][1])):
            pair_dist.append([similar_images[i][1][j],eu_dist(n_features[similar_images[i][0]],n_features[similar_images[i][1][j]])])
        dist_matrix.append([ similar_images[i][0], sorted(pair_dist, key=lambda x: x[1])]) 

    print('\n')
    print("output for 3(c) (Euclidean distances)..............")
    print('\n')

    
    for i in range(len(dist_matrix)):
        print(dist_matrix[i])
        print('\n')
    print('\n')

    print("Solutions for 3(d): extra credit ")
    print('\n')

    reduced_imagesN=intensity_list.map(lambda x:(x[0],small_factor(x[1],5)))

    rN=reduced_imagesN.map(lambda x: (x[0], find_diff(x[1])))

    signaturesN=rN.map(lambda x: (x[0], make_128(x[1])))

    no_of_bands=2
    no_of_buckets=700
    bandsN=signaturesN.map(lambda x: (x[0], divide_into_bands(x[1], no_of_bands)))

    hashed_bandsN=bandsN.flatMap(lambda x:[(x[0], hash_to_buckets(x[1][i], i,no_of_buckets)) for i in range(0,no_of_bands)])

    group_by_bucketsN=hashed_bandsN.map(lambda x: (x[1], x[0])).reduceByKey(lambda x, y: np.append(x,y))

    filter_imagesN=group_by_bucketsN.filter(lambda x: any(item in ('3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19') for item in x[1]))

    zN=filter_imagesN.flatMap(lambda x: [(x[1][i], x[1]) for i in range(len(x[1]))])

    needed_neigboursN=zN.filter(lambda x: x[0] in ('3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19'))

    yN=needed_neigboursN.reduceByKey(lambda x, y: np.append(x,y))


    sN=yN.map(lambda x: (x[0], list(set(x[1]))))

    sample_sizeN=30
    rddsampleN=rN.takeSample(False,sample_sizeN)

    evectorsN=get_svdPlanes(rddsampleN)

    similar_imagesN=sN.filter(lambda x: x[0] in ('3677454_2025195.zip-1', '3677454_2025195.zip-18')).collect()


    lN=[]
    for i in range(len(similar_imagesN)):
        for j in range (len(similar_imagesN[i][1])):
            lN.append(similar_imagesN[i][1][j])

    neighboursN=list(set(lN))


    n_featuresN=rN.filter(lambda x: x[0] in neighboursN).map(lambda x: (x[0], reduce_dimensions(x[1],evectorsN))).collectAsMap()

    dist_matrixN=[]
    for i in range (len(similar_imagesN)):
        pair_distN=[]
        for j in range (len(similar_imagesN[i][1])):
            pair_distN.append([similar_imagesN[i][1][j],eu_dist(n_featuresN[similar_imagesN[i][0]],n_featuresN[similar_imagesN[i][1][j]])])
        dist_matrixN.append([ similar_imagesN[i][0], sorted(pair_distN, key=lambda x: x[1])]) 
    print('\n')
    print("output of 3d... distances with factor 5............")
    print('\n')
    for i in range(len(dist_matrixN)):
        print(dist_matrixN[i])
        print('\n')
    print('\n')


if __name__=="__main__":
    main()
