from imageai.Prediction.Custom import CustomImagePrediction
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

execution_path = os.getcwd()
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("idenprof_061-0.7933.h5")
prediction.setJsonPath("idenprof_model_class.json")
prediction.loadModel(num_objects=10)
i=1
count = int(input("enter number of images to predict on?"))
while i<= count:
    i+=i
    img = input("enter image location?")
    image  = Image.open(img)
# image.show()
    predictions, probabilities = prediction.predictImage(image, result_count=5, input_type="array")
    print (probabilities)
    # labels = predictions
    # sizes = probabilities
    # # colors = ['red', 'yellowgreen', 'lightcoral']
    # # explode = (0.1, 0, 0)  # explode 1st slice
    # #  explode=explode, colors=colors
    # # index = np.arange(len(probabilities))
    # # patches, texts= plt.pie(sizes, shadow=True, startangle=140)
    # plt.bar(predictions,sizes) 
    # # plt.legend(patches, labels, loc="best")
    # # plt.hist(sizes)
    # # plt.axis('equal')
    # # plt.tight_layout()
    # plt.show()
    prob = float(probabilities[0])
    if prob>90:
        desturl= "C:/mini projects/segregation/"+predictions[0]+"/"+img
        image.save(desturl)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction , " : " , eachProbability)  