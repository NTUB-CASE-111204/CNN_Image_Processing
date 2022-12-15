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
'定義子資料夾名稱&對應的數字'
class_names = ['LUSH','DEGUSTER 慢享','Kosmea','純粹森活','Aroma Bella','ARUBLU','Ethique','Ardell','LHAMI','DEEPURE淨森林','Burt’s Bees (Clorox)','Aesop']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (64, 64)
'讀模型'
from keras.models import load_model
model = load_model('12datas_model(300).h5')

from keras.preprocessing import image
import matplotlib.pyplot as plt   
import numpy as np
import tensorflow as tf
picPath = '../CNN/seg_test/DEEPURE/'+'P1040590.jpg'
IMAGE_PATH=picPath #輸入圖片
img=tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(64,64))#跟建模時的input_shape需相同
img=tf.keras.preprocessing.image.img_to_array(img)
plt.imshow(img/255.)
predictions=model.predict(np.array([img]))
#predictions = np.argmax(predictions,axis=1)
print(predictions)
print(class_names[np.argmax(predictions)])
ans = class_names[np.argmax(predictions)]
print(ans)

import psycopg2 
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
conn = psycopg2.connect(host="db.zvkaicfdjrsrevzuzzxh.supabase.co", user="postgres", password ="TiBmTydtbNZ6YfiZ", dbname="postgres")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()
cursor2 = conn.cursor()
print("資料庫連線成功！")
cursor.execute("SELECT b_name FROM public.brand where b_name = '%s'" %(ans))
db = list(cursor.fetchall())
print(db)