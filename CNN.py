import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD, Adam
#%%
'定義子資料夾名稱&對應的數字'
class_names = ['LUSH','DEGUSTER','Kosmea','Purely Life','Aroma Bella','ARUBLU','Ethique','Ardell','LHAMI','DEEPURE']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (64, 64)
#%%
def load_data():
    datasets = ['seg_train', 'seg_test']#資料夾
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #cv讀照片，顏色莫認為BGR，需轉為RGB，錯誤表示黑白或已轉
                image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output
#%%
(train_images, train_labels), (test_images, test_labels) = load_data()
#%%
'隨機性'
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
'標準化'
train_images = train_images / 255.0 
test_images = test_images / 255.0
#%%
'建模'
input_shape = (64, 64, 3)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu', strides=2),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.2),
    Conv2D(128, (3, 3), input_shape=input_shape, padding='same',
           activation='relu', strides=2),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.2),
    Flatten(),
    Dropout(0.5),
    Dense(10, activation='softmax') #輸出層，分類用softmax，多少組照片就用多少的數字
])
model.compile(optimizer = 'adam', #SGD(lr=0.1)
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
#%%
history = model.fit(train_images, train_labels, 
                    #validation_data=(test_images, test_labels),
                    #verbose=2,callbacks=[earlyStop],
                    batch_size=128, epochs=200)
#%%
'模型概況'
plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot(history.history["loss"])
#scores = model.evaluate(test_images, test_labels)  
#print('test:',result[1])
#%%
'預測'
predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
#%%
'混淆矩陣'
CM = confusion_matrix(test_labels, pred_labels)
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 
print(accuracy(CM))
#%%
'混淆矩陣視覺化，看錯誤'
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()
#%%
'存模型'
from keras.models import load_model
model.save("10datas_model(fail5).h5")
#%%
'讀模型'
from keras.models import load_model
model = load_model('100%_model')

from keras.preprocessing import image
import matplotlib.pyplot as plt   
import numpy as np
import tensorflow as tf
IMAGE_PATH='C:/chashin/2技/python文字辨識/goro.jpg' #輸入圖片
img=tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(64,64))#跟建模時的input_shape需相同
img=tf.keras.preprocessing.image.img_to_array(img)
plt.imshow(img/255.)
predictions=model.predict(np.array([img]))
#predictions = np.argmax(predictions,axis=1)
print(predictions)
print(class_names[np.argmax(predictions)])
ans = class_names[np.argmax(predictions)]
print(ans)
'''
#%%
from keras.applications.xception import xception,decode_predictions
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt   
#%%
model=tf.keras.applications.xception.Xception(weights='imagenet',include_top=True)

#%%
'任意照片&格式轉換'
IMAGE_PATH='C:/chashin/2技/python文字辨識/goro.jpg'
img=tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(299,299))
img=tf.keras.preprocessing.image.img_to_array(img)
plt.imshow(img/255.)

#%%
'辨識'
predictions=model.predict(np.array([img]))
print('Predicted:', decode_predictions(predictions, top=3)[0])
'''