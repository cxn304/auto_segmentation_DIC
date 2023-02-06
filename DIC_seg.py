# -*- coding: utf-8 -*-
'''
author: cxn
version: 0.1.1
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpltPath
import tensorflow as tf
from tensorflow import keras
import cv2, time, pyclipper
import os
# from skimage import morphology
from scipy.cluster.vq import kmeans,vq,whiten

class TextImageGenerator:
    def __init__(self, train_dir, test_dir, img_size, img_name, predictma, num_channels=1):
        self._train_dir = train_dir
        self._img_name = img_name
        self._test_dir = test_dir
        self._num_channels = num_channels
        self.img_w, self._img_h = img_size
        self._predictma = predictma

        self._num_examples = 0
        self._test_num_examples = 0
        self.filenames = []
        self.test_filenames = []
        self.labels = []
        self.test_labels = []

        self.init()

    def init(self):
        if self._predictma == False:
            self.__train_image_cut()
        else:
            self.__predict_image_cut50()
            
                    
    def __train_image_cut(self):
        fs = os.listdir(self._train_dir)
        for filename in fs:
            self.filenames.append(filename)
        for filename in self.filenames:
            tempLable=filename.split("_",1)
            label = tempLable[1]
            label = label[:-4]
            self.labels.append(label)
            self._num_examples += 1
        self.labels = np.float32(self.labels)

        testfs = os.listdir(self._test_dir)
        for filename in testfs:
            self.test_filenames.append(filename)
        for filename in self.test_filenames:
            tempLable=filename.split("_",1)
            label = tempLable[1]
            label = label[:-4]
            self.test_labels.append(label)
            self._test_num_examples += 1
        self.test_labels = np.float32(self.test_labels)
        
        perm = np.arange(self._num_examples)  
        np.random.shuffle(perm)           
        self._filenames = [self.filenames[i] for i in perm] 
        self._labels = self.labels[perm]    
        self.x_train = np.zeros([self._num_examples,self.img_w, self._img_h,1])
        for i in range(self._num_examples):
            temp_image = cv2.imread(os.path.join(self._train_dir, self._filenames[i])
                                    ,cv2.IMREAD_GRAYSCALE)
            temp_image = temp_image/255.0 
            self.x_train[i,:,:,0] = temp_image 
        self.y_train = self._labels
        
        perm = np.arange(self._test_num_examples)  
        np.random.shuffle(perm)     
        self._test_filenames = [self.test_filenames[i] for i in perm] 
        self._test_labels = self.test_labels[perm] 
        self.x_test = np.zeros([self._test_num_examples,self.img_w, self._img_h,1])
        for i in range(self._test_num_examples):
            temp_image = cv2.imread(os.path.join(self._test_dir, self._test_filenames[i])
                                    ,cv2.IMREAD_GRAYSCALE)
            temp_image = temp_image/255.0 
            self.x_test[i,:,:,0] = temp_image 
        self.y_test = self._test_labels
    
    
    def __predict_image_cut50(self):
        temp_image = cv2.imread(self._img_name
                                ,cv2.IMREAD_GRAYSCALE)
        hang,lie = temp_image.shape
        self.hang=hang-hang%self._img_h 
        self.lie=lie-lie%self._img_h
        self.steps = 50
        n = 0
        self.x_predict = np.zeros([(self.hang//self.steps-1)*(self.lie//self.steps-1),
                                   self.img_w, self._img_h,1])
        self.new_image = temp_image[0:self.hang,0:self.lie]
        for i in range(0,self.hang-self.steps,self.steps):
            for j in range(0,self.lie-self.steps,self.steps):
                cut_image = self.new_image[i:i+self._img_h,j:j+self._img_h]
                cut_image = cut_image/255.0
                self.x_predict[n,:,:,0] = cut_image
                n = n+1

                
    def __predict_image_cut100(self):
        temp_image = cv2.imread(self._img_name
                                ,cv2.IMREAD_GRAYSCALE)
        hang,lie = temp_image.shape
        self.hang=hang-hang%self._img_h 
        self.lie=lie-lie%self._img_h
        self.steps = 100
        n = 0
        self.x_predict = np.zeros([(self.hang//self.steps)*(self.lie//self.steps),
                                   self.img_w, self._img_h,1])
        self.new_image = temp_image[0:self.hang,0:self.lie]
        for i in range(0,self.hang,self.steps):
            for j in range(0,self.lie,self.steps):
                cut_image = self.new_image[i:i+self._img_h,j:j+self._img_h]
                cut_image = cut_image/255.0
                self.x_predict[n,:,:,0] = cut_image
                n = n+1
        
        
        
class SoftmaxModel(tf.keras.Model): 
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu') 
        self.flatten = keras.layers.Flatten()  
        self.d1 = keras.layers.Dense(128, activation='relu') 
        self.d2 = keras.layers.Dense(4, activation='softmax') 

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)



class SigmoidModel(tf.keras.Model): 

    def __init__(self):
        super(SigmoidModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu') 
        self.mx1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
        self.flatten = keras.layers.Flatten()  
        self.d1 = keras.layers.Dense(128, activation='relu')
        self.d2 = keras.layers.Dense(2, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.mx1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x) 
    
  
def DIC_train_predict(types, predictable, isSigmoid):

    if (predictable == False):
        if types == "softmax":
            train_ds = tf.data.Dataset.from_tensor_slices((train_data.x_train,
                                                           train_data.y_train)).batch(32)
            test_ds = tf.data.Dataset.from_tensor_slices((train_data.x_test, 
                                                              train_data.y_test)).batch(32)
            model = SoftmaxModel()
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy()   

            optimizer = tf.keras.optimizers.Adam() 

            train_loss = tf.keras.metrics.Mean(name='train_loss')

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            test_loss = tf.keras.metrics.Mean(name='test_loss')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        else:
            temp_ytrain = (train_data.y_train+1)%2  
            temp_ytrain=np.column_stack((temp_ytrain,train_data.y_train))   
            train_ds = tf.data.Dataset.from_tensor_slices((train_data.x_train,
                                                           temp_ytrain)).batch(32)
            temp_ytest = (train_data.y_test+1)%2
            temp_ytest=np.column_stack((temp_ytest,train_data.y_test))
            test_ds = tf.data.Dataset.from_tensor_slices((train_data.x_test, 
                                                          temp_ytest)).batch(32)
            model = SigmoidModel()
            loss_object = tf.keras.losses.BinaryCrossentropy()   

            optimizer = tf.keras.optimizers.Adam()
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
            test_loss = tf.keras.metrics.Mean(name='test_loss')
            test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
            
        
        @tf.function
        def train_step(images, labels): 
            with tf.GradientTape() as tape: 
                predictions = model(images) 
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
            train_loss(loss)
            train_accuracy(labels, predictions)
            
        @tf.function
        def test_step(images, labels): 
            predictions = model(images)
            t_loss = loss_object(labels, predictions) 
            test_loss(t_loss)
            test_accuracy(labels, predictions)
        
        EPOCHS = 100  
        for epoch in range(EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            
            for images, labels in train_ds:
                train_step(images, labels)
        
            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
        
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print (template.format(epoch+1,
                                   train_loss.result(),
                                   train_accuracy.result()*100,
                                   test_loss.result(),
                                   test_accuracy.result()*100))
        
        
        if(isSigmoid):
            model.save(r'.\sigmoid_checkpoints\sigmoid_model',save_format='tf')
        else:
            model.save(r'.\softmax_checkpoints\softmax_model',save_format='tf')
        return "train done"
        
    else:

        if(isSigmoid):
            model = SigmoidModel()
            model.load_weights('./sigmoid_weights')
            start_time = time.time()
            probability_model = keras.Sequential([model])
            predictions = probability_model.predict(train_data.x_predict)   
            predict_time = (time.time() - start_time)
        else:
            model = SoftmaxModel()
            model.load_weights('./softmax_weights')
            start_time = time.time()
            probability_model = keras.Sequential([model])
            predictions = probability_model.predict(train_data.x_predict)
            predict_time = (time.time() - start_time)
        
        return predictions,predict_time
        

def plot_in_figure(predict_finals, sigmoidf):
    if sigmoidf == False:
        cd = len(predict_finals)
        plt.figure()
        plt.imshow(train_data.new_image, cmap=plt.cm.gray,dpi=100)
        for i in range(cd):
            if np.argmax(predict_finals[i])==0:
                h = i//(train_data.lie//100)
                l = i%(train_data.lie//100) 
                ax = plt.gca()
                rect = patches.Rectangle((l*100,h*100),100,100,linewidth=1,
                                         edgecolor='m',facecolor='none')
                ax.add_patch(rect)
            elif np.argmax(predict_finals[i])==1:
                h = i//(train_data.lie//100)
                l = i%(train_data.lie//100) 
                ax = plt.gca()
                rect = patches.Rectangle((l*100,h*100),100,100,linewidth=1,
                                         edgecolor='b',facecolor='none')
                ax.add_patch(rect)
            elif np.argmax(predict_finals[i])==2:
                h = i//(train_data.lie//100)
                l = i%(train_data.lie//100) 
                ax = plt.gca()
                rect = patches.Rectangle((l*100,h*100),100,100,linewidth=1,
                                         edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            elif np.argmax(predict_finals[i])==3:
                h = i//(train_data.lie//100)
                l = i%(train_data.lie//100) 
                ax = plt.gca()
                rect = patches.Rectangle((l*100,h*100),100,100,linewidth=1,
                                         edgecolor='c',facecolor='none')
                ax.add_patch(rect)
        plt.show()

    else:
        trimap = train_data.new_image.copy()
        trimap[:] = 0
        predict_finals = predict_finals.reshape(train_data.hang//train_data.steps-1, 
                                                train_data.lie//train_data.steps-1,2)  
        likelihood = predict_finals[:,:,1]
        likelihoods = likelihood.copy()
        likelihoods[likelihood < 0.95] = 0
        likelihoods[likelihood >= 0.95] = 1
        likelihoods = np.where(likelihoods,likelihoods,np.nan)
        height,width = trimap.shape
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.axis('off')
        plt.imshow(train_data.new_image, cmap=plt.cm.gray)
        outer = []
        inner = []
        
        for i in range(train_data.hang//train_data.steps-1):
            for j in range(train_data.lie//train_data.steps-1):
                if(likelihoods[i,j] == 1):
                    ax = plt.gca()
                    rect = patches.Rectangle((j*train_data.steps,i*train_data.steps)
                                             ,train_data.img_w,train_data.img_w,linewidth=2,
                                         edgecolor='b',facecolor='none')
                    ax.add_patch(rect)
                    trimap[i*train_data.steps:i*train_data.steps+train_data.img_w,
                           j*train_data.steps:j*train_data.steps+train_data.img_w] = 1
                    inner.append(train_data.new_image[i*train_data.steps:i*train_data.steps+train_data.img_w,
                           j*train_data.steps:j*train_data.steps+train_data.img_w])
                else:
                    ax = plt.gca()
                    rect = patches.Rectangle((j*train_data.steps,i*train_data.steps)
                                             ,train_data.img_w,train_data.img_w,linewidth=0.5,
                                         edgecolor='m',facecolor='none')
                    ax.add_patch(rect)
                    outer.append(train_data.new_image[i*train_data.steps:i*train_data.steps+train_data.img_w,
                           j*train_data.steps:j*train_data.steps+train_data.img_w])
    return trimap, likelihoods,outer,inner


def find_max_CC(bw_img,N):
    if N>1:
        d = 3
        while True:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (d,d))
            d=d+1
            bw_img_return = cv2.erode(bw_img, kernel, iterations=1)
            num_labels,labels,stats,_=cv2.connectedComponentsWithStats(bw_img_return)
            b=np.array([np.arange(num_labels)]).T
            stats_sort = np.concatenate((b, stats), axis=1)
            stats_sort = stats_sort[stats_sort[:,5].argsort()] 
            stats_sort = stats_sort[0:-1,:]
            centroids, _ = kmeans(np.float32(stats_sort[:,5]),N+1)
            centroids = centroids[centroids.argsort()]
            if centroids[-1]/centroids[-2]<3 and centroids[-2]/centroids[0]>10:
                zuida_id = stats_sort[-1,0] 
                z_c_c = np.uint8(np.zeros(bw_img_return.shape))
                z_c_c[labels==zuida_id]=1 
                z_c_c = cv2.dilate(z_c_c, kernel, iterations=1)
                return z_c_c,stats_sort
            
    else:
        num_labels,labels,stats,_=cv2.connectedComponentsWithStats(bw_img)
        b=np.array([np.arange(num_labels)]).T
        stats_sort = np.concatenate((b, stats), axis=1)
        stats_sort = stats_sort[stats_sort[:,5].argsort()] 
        stats_sort = stats_sort[0:-1,:]
        zuida_id = stats_sort[-1,0] 
        z_c_c = np.uint8(np.zeros(bw_img.shape))
        z_c_c[labels==zuida_id]=1
        return z_c_c,stats_sort

def find_mean_diameter(bw2_img):
    num_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(bw2_img)
    b=np.array([np.arange(num_labels)]).T
    stats_sort = np.concatenate((b, stats), axis=1)
    stats_sort = stats_sort[stats_sort[:,5].argsort()] 
    for i in range(len(stats_sort)):
        if stats_sort[i,5]>=9: 
            stats_after_delete=stats_sort[i:-2,:]
            break
    kernel_size = stats_after_delete[:,5] 
    kernel_size=np.float32(kernel_size)
    centroids, _ = kmeans(kernel_size,6)
    each_point_cluster_num, dist = vq(kernel_size, centroids)
    six_num = []
    for i in range(6):
        six_num.append(np.sum(each_point_cluster_num == i))
    most_kernel_size = six_num.index(max(six_num))
    most_kernel_size_index = np.where(each_point_cluster_num==most_kernel_size)[0]
    mean_diameter = int(np.average(kernel_size[most_kernel_size_index]))
    return mean_diameter


def delete_ostihole(da_lty,mean_diameter):
    lty_fan = ~da_lty
    d = mean_diameter
    num_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(lty_fan)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (d,d))
    while min(stats[:,4])<(mean_diameter*3)**2:
        d = int(d*1.2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (d,d))
        da_lty=cv2.morphologyEx(da_lty,cv2.MORPH_CLOSE,kernel)
        lty_fan = ~da_lty
        num_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(lty_fan)
    da_lty=cv2.morphologyEx(da_lty,cv2.MORPH_OPEN,kernel)
    return da_lty
    

def morph_img(gray_img,N):
    if len(gray_img.shape)==3:
        gray_img = gray_img[:,:,0]
    thres,bw2_img=cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    mean_diameter = find_mean_diameter(~bw2_img)
    dd = int(mean_diameter*0.4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dd,dd))
    bw2_canny = cv2.Canny(gray_img, 170, 260)
    bw2_canny_done = cv2.morphologyEx(bw2_canny,cv2.MORPH_CLOSE,kernel)
    z_c_c,stats_sort = find_max_CC(bw2_canny_done,N)
    if N==2:
        sx_lty = bw2_canny_done-z_c_c*255
        sx_lty_close = cv2.morphologyEx(sx_lty,cv2.MORPH_CLOSE,kernel)
        
        er_c_c,stats_sort_sx = find_max_CC(sx_lty_close,1)
    
        com_c_c = sx_lty_close-er_c_c*255
        com_c_c = cv2.morphologyEx(com_c_c,cv2.MORPH_OPEN,kernel)
        z_c_c = z_c_c*255+com_c_c
        z_c_c_close = cv2.morphologyEx(z_c_c,cv2.MORPH_CLOSE,kernel)
        z_c_c_close,_ = find_max_CC(z_c_c,1)
        z_c_c_close=delete_ostihole(z_c_c_close*255,mean_diameter)
        
        er_c_c=delete_ostihole(er_c_c*255,mean_diameter)
        return z_c_c_close,er_c_c,bw2_canny,bw2_canny_done
    elif N==1:
        z_c_c_close=delete_ostihole(z_c_c*255,mean_diameter)
        return z_c_c_close,N,bw2_canny,bw2_canny_done


def plot_contours(gray_img,dy_c_c,der_c_c,bw2_canny,bw2_canny_done,N):
    dycontours,hierarchy=cv2.findContours(
        dy_c_c,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if N==2:
        dercontours,hierarchy=cv2.findContours(
            der_c_c,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        C=dycontours+dercontours
        ff=cv2.drawContours(gray_img, C, -1, (0,255,0), 3)
        plt.rcParams.update({"font.size":20})
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(ff)
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(dy_c_c)
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(der_c_c)
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(bw2_canny_done)
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(bw2_canny)
    elif N==1:
        C=dycontours
        ff=cv2.drawContours(gray_img, C, -1, (0,255,0), 3)
        plt.rcParams.update({"font.size":20})
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(ff)
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(dy_c_c)
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(bw2_canny_done)
        plt.figure(figsize=(height/100,width/100),dpi=150)
        plt.imshow(bw2_canny)
    
 
plt.rcParams['savefig.dpi'] = 200 
plt.rcParams['figure.dpi'] = 200   

is_predict = True
is_sigmoid = True
if(is_sigmoid):
    trainDir = r'.\train_img_sigmoid'
    testDir = r'.\test_img_sigmoid'
    sort_type = "sigmoid"
else:
    trainDir = r'.\train_img_softmax'
    testDir = r'.\test_img_softmax'
    sort_type = "softmax"

image_name = './work_img/MST_2.png'
NN = 1
train_data=TextImageGenerator(trainDir, testDir,
                              [100,100], image_name, is_predict) 
predict_final,predict_time = DIC_train_predict(sort_type, is_predict, is_sigmoid)

if_dic_pixel,predict_logic,outer,inner = plot_in_figure(predict_final, is_sigmoid)
predict_logic_full = np.where(if_dic_pixel,if_dic_pixel,np.nan)  
thres_outer = np.mean(outer)
thres_inner = np.mean(inner)
no_disturb_gimage = np.uint8(
    train_data.new_image*if_dic_pixel+~(if_dic_pixel*255)/255*int(thres_outer))

dy_c_c,der_c_c,bw2_canny,bw2_canny_done = morph_img(no_disturb_gimage,NN)
height,width = dy_c_c.shape

plot_contours(cv2.imread(image_name),
            dy_c_c,der_c_c,bw2_canny,bw2_canny_done,NN)

            
            
            
