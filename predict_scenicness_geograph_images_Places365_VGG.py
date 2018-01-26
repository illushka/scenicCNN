
####################################################################################
# Code for using Scenic CNN to predict the scenicness of outdoor images on Geograph http://www.geograph.org.uk/
# Code by Chanuki Seresinhe c.seresinhe@warwick.ac.uk
# Email Chanuki Seresinhe to get access to Scenic CNN Caffe Model
# Some code snippets have been taken from from http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
# Paper Reference:
# Seresinhe, C. I., Preis, T., & Moat, H. S. (2017). Using deep learning to quantify the beauty of outdoor places. Royal Society Open Science, 4, 170170.
# http://rsos.royalsocietypublishing.org/content/4/7/170170

####################################################################################
# 1. General setup
####################################################################################


import os
import numpy as np
import sys
import scipy.io as sio
import pandas as pd

import urllib
import math
import time



#Specify model name and parameters
model_name = "model9-vgg"
model_folder = model_name +"/"



run_azure = 1


gpu_id = int(sys.argv[1])  # comment for test
start_num = int(sys.argv[2]) # comment for test
last_num = int(sys.argv[3]) # comment for test


#gpu_id = 0
#start_num = 0
#last_num = 100


if run_azure:
    test_data = "azure_test.txt"

    caffe_root = "where is caffe?"
    code_folder = "where is code?"
    folder = "/home/chanuki/storage_disk/geograph_img/"

    import sys
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    print("running on cluster...")


else:
    test_data = "test.txt"
    caffe_root = "where is caffe?"
    code_folder = "where is code?"
    folder = code_folder + "geograph_img/"

    sys.path.insert(0, caffe_root + 'python')
    import caffe
    print(test_data)


#How freuqently do you want to save the file
counter_num = 5000


#Ensure you have already removed any indoor images from dataset
geograph_data = pd.read_csv(code_folder + "data/geograph_london_ONLY_OUT.csv")

#You will have to request the hashes from Geograph diretly
image_data = pd.read_table(code_folder + 'data/london-hashes.tsv', header=0)


max_run=len(geograph_data)

#max_run=20 #use for testing

geograph_data["Predicted_Score"] = ""


#####################################################################################


model_def = code_folder + "results/transfer_learning/"+ model_folder +"scenic_CNN.prototxt"
model_weights = code_folder + "results/transfer_learning/" + model_folder + "weights_" +model_name + ".caffemodel"

labels_file = caffe_root + 'models/places365/IO_places365.csv'

mean_file = caffe_root + 'models/places365/places365CNN_mean.npy'

if not os.path.exists(mean_file):
        print("creating mean file..")
        blob = caffe_pb2.BlobProto()
        with open(mean_file .rstrip("npy")+"binaryproto", "rb") as fp:
            blob.ParseFromString(fp.read())
        np.save(caffe_root + "models/places365/places365CNN_mean.npy", blobproto_to_array(blob))


mean_image = np.load(mean_file).reshape((3, 256, 256)).mean(1).mean(1)


net = caffe.Classifier(model_def,
                        model_weights,
                        mean=mean_image,
                        channel_swap=(2, 1, 0),
                        raw_scale=255,
                        image_dims=(256, 256))


#Choose size
size="small"

def getGeographUrl(gridimage_id, image_hash, size):

    yz= str(int(math.floor(gridimage_id/1000000))).zfill(2)
    ab=str(int(math.floor((gridimage_id%1000000)/10000))).zfill(2)
    cd=str(int(math.floor((gridimage_id%10000)/100))).zfill(2)
    abcdef=str(gridimage_id).zfill(6)

    if (yz == '00'):
        fullpath="/photos/"+ab+"/"+cd+"/"+abcdef+"_"+image_hash;
    else:
        fullpath="/geophotos/"+yz+"/"+ab+"/"+cd+"/"+abcdef+"_"+image_hash;

    if (size=="medium"):
        return "http://s"+str(gridimage_id%4)+".geograph.org.uk"+fullpath+"_213x160.jpg"

    else:
        return "http://s0.geograph.org.uk"+fullpath+".jpg"




def imageAnalyse(path, geograph_data, df_id):

    print("analysing image")
    image = caffe.io.load_image(path)

    prediction = net.predict([image], oversample=True)

    predicted_score = prediction[0][0]

    #print("Predicted score")
    #print( predicted_score)


    #Store classification back into dataframe
    geograph_data.set_value(df_id, 'Predicted_Score', predicted_score)

    return geograph_data

counter = 1

for i in range(start_num, last_num):
    print(i)
    gridimage_id = geograph_data.iloc[i]['gridimage_id']
    df_id=i
    # print gridimage

    # get hash number
    row = image_data[image_data.gridimage_id == int(gridimage_id)]

    #Savefile intermittently so you don't lose too much data if somethng goes wrong
    if len(row) > 0:
        counter += 1
        if (counter % counter_num == 0):

            geograph_data_tosave = geograph_data
            geograph_data_tosave.to_csv(code_folder + "results/transfer_learning/" + model_folder +'geograph_data_predictions_RESNET_'+str(gpu_id)+ '_' + str(counter) + '_1strun.csv')
            print(os.getcwd())
            print("saving file...")

            # delete file
            try:
                os.remove(code_folder + "results/transfer_learning/" + model_folder +'geograph_data_predictions_RESNET_' + str(counter - counter_num) + '.csv')
            except:
                print 'error removing file'

        if (counter == max_run):
            break

        image_hash = row.iloc[0]['hash']

        path = folder + str(gridimage_id) + ".jpg"
        image_URL = getGeographUrl(int(gridimage_id), image_hash, size)  #
        #print("first df_id")
        #print(df_id)
        print(path)

        if not os.path.isfile(path):


            time.sleep(1)

            try:
                #print(path)
                #print(image_URL)
                urllib.urlretrieve(image_URL, path)
                print "downloading image"

            except:
                print "image failed 1"

        if os.path.isfile(path):
            #geograph_data = imageAnalyse(path, geograph_data, df_id) #use for testing
            try:
                geograph_data = imageAnalyse(path, geograph_data, df_id)


            except:
                continue

        else:
            print "image failed 2"


geograph_data.to_csv(code_folder + "results/transfer_learning/" + model_folder + "geograph_data_predictions_VGG"+str(gpu_id)+"_1strun.csv")

