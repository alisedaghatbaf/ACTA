# example of training an conditional gan on the fashion mnist dataset
import statistics
from numpy import expand_dims
from numpy import squeeze
from numpy import zeros
from numpy import ones
from numpy import copy
from numpy import count_nonzero
from numpy import asarray
from numpy import round
from numpy import append
from numpy import array
from numpy import unique
from numpy import ndarray
from numpy import trace
from numpy import concatenate
from numpy import vstack
from numpy import iscomplexobj
from numpy import cov

from scipy.linalg import sqrtm
from numpy import argpartition
from numpy.random import randn
from numpy.random import randint
from numpy.random import choice
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv1DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from matplotlib import pyplot
from numpy import float32
from pandas import DataFrame
from pandas import read_csv
from pandas import concat
from skimage.transform import resize
import matplotlib.pyplot as plt
from IPython.display import clear_output
    

def define_discriminator(in_shape, n_classes): 
   
    in_label = Input(shape=(1,)) 
    
    li = Embedding(n_classes, 10)(in_label) 
    n_nodes = in_shape[0] 
    li = Dense(n_nodes)(li)
    
    
    li = Reshape((n_nodes, 1))(li) 
   
    in_data = Input(shape=in_shape)
    
    merge = Concatenate()([in_data, li])
    
    hidden1 = Dense(128)(merge)
    hidden1 = Activation('relu')(hidden1)
    hidden2 = Dense(128)(hidden1)
    hidden2 = Activation('relu')(hidden2)
    
    hidden3 = Flatten()(hidden2)
    
    out_layer = Dense(1, activation='sigmoid')(hidden3)
    
    model = Model([in_data, in_label], out_layer)
    
    opt = Adam(lr=0.0001, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])    
   
    return model

def define_generator(latent_dim, input_dim, n_classes):
    
    in_label = Input(shape=(1,))    
    li = Embedding(n_classes, 10)(in_label)     
    n_nodes = input_dim
    li = Dense(n_nodes)(li)    
    li = Reshape((n_nodes, 1))(li) 
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 128 * input_dim
    gen = Dense(n_nodes)(in_lat)
    gen = Reshape((input_dim, 128))(gen)
   
    merge = Concatenate()([gen, li])
    hidden1 = Dense(128)(merge)
    hidden1 = Activation('relu')(hidden1)
    hidden2 = Dense(128)(hidden1)
    hidden2 = Activation('relu')(hidden2)
    
    out_layer = Dense(1, activation='tanh')(hidden2)
    
    model = Model([in_lat, in_label], out_layer)
    
    return model

def define_gan(g_model, d_model):
    
    d_model.trainable = False
    
    gen_noise, gen_label = g_model.input
   
    gen_output = g_model.output
    
    gan_output = d_model([gen_output, gen_label])
    
    model = Model([gen_noise, gen_label], gan_output)
    
    opt = Adam(lr=0.0001, beta_1=0.5)
    
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def filter(X, y , clusters):
    n_samples = len(y)
    for i in range(n_samples):        
        x = X[i]
        cid = x[0]
        rid = x[1]
        iid = x[2]
        uid = x[3]
        for j in range(len(clusters)):
            cl = clusters[j]
            if cid <= cl[1][1] and cid >= cl[1][0]:
                if rid <= cl[2][1] and rid >= cl[2][0]:
                    if iid <= cl[3][1] and iid >= cl[3][0]:
                        if uid <= cl[4][1] and uid >= cl[4][0]:                            
                            y[i]= 0
                            break
    return X, y


def load_real_samples(path, n_samples, filter_clusters = None):
    df = read_csv(path)
    data = df.to_numpy()
    n= min(n_samples, len(data))
    y = data[:n, -1]
    X = data[:n, 1:-1]
    if filter_clusters is not None:
        X, y = filter(X, y, filter_clusters)
    X = X.astype('float32')
    
    X = (X - 31) / 31
    print(X.shape, y.shape)
    return [X, y]

def label_fake_samples(X):
    clusters = read_csv('gan_clusters.csv').values.tolist()
    for j in range(len(clusters)):
            cl = clusters[j]            
            for k in range(1,5):
                clstr = cl[k][1:-1]                
                cl[k] = tuple(map(int, clstr.split(', '))) 
    n_samples = len(X)
    l = zeros(n_samples) #ndarray((n_samples, n_classes), dtype = int)
    X = round(X*31+31)
    for i in range(n_samples):
        
        x = X[i]
        cid = x[0]
        rid = x[1]
        iid = x[2]
        uid = x[3]
        for j in range(len(clusters)):
            cl = clusters[j]
            if cid <= cl[1][1] and cid >= cl[1][0]:
                if rid <= cl[2][1] and rid >= cl[2][0]:
                    if iid <= cl[3][1] and iid >= cl[3][0]:
                        if uid <= cl[4][1] and uid >= cl[4][0]:                            
                            l[i]= 1
                            break
    return l

def fake_to_real(X_fake, y_fake, budget):    
    
    X_real = squeeze(X_fake, axis=-1)
    y_real =  squeeze(y_fake, axis=-1)    
    y_real = abs(y_real - 0.5)
    idx = argpartition(y_real, budget-1)[:budget]
    X_real = X_real[idx]
    label_real = label_fake_samples(X_real)         
    
    return idx, label_real


def select_real_samples(dataset, n_samples):
    
    data, labels = dataset    
    ix = randint(0, data.shape[0], n_samples)    
    
    X, labels = data[ix], labels[ix]
    
    X = expand_dims(X, axis=-1)
    y = ones((n_samples, 1))
    return [X, labels], y


def generate_latent_points(latent_dim, n_samples, n_classes):
   
    x_input = randn(latent_dim * n_samples)
    
    z_input = x_input.reshape(n_samples, latent_dim)
    
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def generate_fake_samples(generator, latent_dim, n_samples, n_classes):
    
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
   
    data = generator.predict([z_input, labels_input])
    
    y = zeros((n_samples, 1))
    return [data, labels_input], y


def scale_data(data, new_shape):
    data_list = list()
    for datom in data:
        
        new_datom = resize(datom, new_shape, 0)
        
        data_list.append(new_datom)
    return asarray(data_list)
 

def calculate_fid(model, data1, data2, labels1, labels2):
    
    act1 = model.predict(data1)
    act2 = model.predict(data2)
    act1 = concatenate([act1, labels1], axis=1)
    act2 = concatenate([act2, labels2], axis=1)
    
    
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    ssdiff = sum((mu1 - mu2)**2.0)
    
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode, filters=None, n_epochs=100, n_batch=64):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    fid = 100.0
    #inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # manually enumerate epochs
    valids = list()
    maxes = list()
    means = list()
    sdvs = list()
    max_valid = 0
    for i in range(n_epochs):
        if max_valid > 190:
            break
        
        for j in range(bat_per_epo):
            
            
            [X_real, labels_real], y_real = select_real_samples(dataset, half_batch)
               
            d_loss1, d_acc1 = d_model.train_on_batch([X_real, labels_real], y_real)        
            
            
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, n_classes)
            
            if active_mode:
                prediction = d_model.predict([X_fake, labels])
                idx_active, labels_active = fake_to_real(X_fake, prediction, int(half_batch/2))                
                y_fake[idx_active] = 1  
                labels[idx_active] = labels_active              
                             
            
            d_loss2, d_acc2 = d_model.train_on_batch([X_fake, labels], y_fake)    
    
            
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch, n_classes)
            
            y_gan = ones((n_batch, 1))
            
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            
            n_good_samples = 200
            latent_points, labels = generate_latent_points(latent_dim, n_good_samples, n_classes)
            
            labels = zeros([n_good_samples, 1], dtype=int) #zeros([n_samples, n_classes], dtype=int)
            labels[:, 0] = 1
            
            X = g_model.predict([latent_points, labels])
            X = round(X*31+31)
            results = DataFrame(X.reshape((X.shape[0], X.shape[1]))).drop_duplicates(ignore_index=True)   
            clusters = read_csv('cgan_clusters.csv').values.tolist()
                
            v = check_validity(results, clusters, filters)
            if v >= max_valid:
                
                max_valid = v
                g_model.save('cgan_generator.h5')
                d_model.save('cgan_discriminator.h5')
               
            valids.append(v) 
            maxes.append(max_valid)
            m = v
            stdv = 0
            if j > 1:
                m = statistics.mean(valids)
                stdv = statistics.stdev(valids)
            means.append(m)
            sdvs.append(stdv)
            
            print('\r', i+1, j+1, 'valid items:', v, ' max valids:', max_valid, ' mean valids:', m, ' stdv valids:', stdv, end='', flush=True)
            
            if max_valid > 190:
                break        
        
    results = DataFrame(columns=['max','mean', 'sdv'])        
    results['max'] = maxes
    results['mean'] = means
    results['sdv'] = sdvs
    results.to_csv('cgan_results.csv')
    
def generate_good_samples( latent_dim, n_classes, n_samples, idx_class):
    model = load_model('cgan_generator.h5', compile=False)
   
    latent_points, labels = generate_latent_points(latent_dim, n_samples, n_classes)
    
    labels = zeros([n_samples, 1], dtype=int) 
    labels[:, 0] = idx_class
    
    X = model.predict([latent_points, labels])
    X = round(X*31+31)
    return X
    
def init_model(latent_dim, input_dim, n_classes, n_samples, active_mode, data_file):
    
    d_model = define_discriminator((input_dim, 1), n_classes)
    
    g_model = define_generator(latent_dim, input_dim, n_classes)
    
    gan_model = define_gan(g_model, d_model)
    
    dataset = load_real_samples(data_file, n_samples)      
    
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode)    
    filters = [[0,(5,14),(6,21),(45,45),(12,27)],[6,(6,9),(41,56),(7,22),(5,20)]]
    dataset = load_real_samples(data_file, n_samples, filters)      
    
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode, filters)  
    filters.extend([[2,(6,21),(14,29),(27,37),(9,19)],[3,(12,13),(10,25),(27,42),(11,26)],[7,(16,18),(27,42),(10,25),(9,16)]])
    dataset = load_real_samples(data_file, n_samples, filters)      
    
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode, filters)  
    filters.extend([[5,(6,21),(29,44),(3,18),(15,30)],[4,(9,15),(26,41),(4,19),(3,8)],[1,(3,5),(11,26), (1,8),(1,16)], [8,(4,19),(25,40),(38,43),(41,48)]])
    dataset = load_real_samples(data_file, n_samples, filters)      
   
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode, filters)  

def update_model(latent_dim, input_dim, n_classes, n_samples, active_mode, data_file):
    
    d_model = load_model('cgan_discriminator.h5')
    opt = Adam(lr=0.0001, beta_1=0.5)
    
    d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])    
    
    g_model = load_model('cgan_generator.h5')
   
    gan_model = define_gan(g_model, d_model)
    
    dataset = load_real_samples(data_file, n_samples)
    
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode)    
    filters = [[0,(5,14),(6,21),(45,45),(12,27)],[6,(6,9),(41,56),(7,22),(5,20)]]
    dataset = load_real_samples(data_file, n_samples, filters)      
    
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode, filters)  
    filters.extend([[2,(6,21),(14,29),(27,37),(9,19)],[3,(12,13),(10,25),(27,42),(11,26)],[7,(16,18),(27,42),(10,25),(9,16)]])
    dataset = load_real_samples(data_file, n_samples, filters)      
    
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode, filters)  
    filters.extend([[5,(6,21),(29,44),(3,18),(15,30)],[4,(9,15),(26,41),(4,19),(3,8)],[1,(3,5),(11,26), (1,8),(1,16)], [8,(4,19),(25,40),(38,43),(41,48)]])
    dataset = load_real_samples(data_file, n_samples, filters)      
    
    train(g_model, d_model, gan_model, dataset, latent_dim, n_classes, active_mode, filters)  
    
def check_validity(results, clusters, filters):
    for j in range(len(clusters)):
            cl = clusters[j]            
            for k in range(1,5):
                clstr = cl[k][1:-1]                
                cl[k] = tuple(map(int, clstr.split(', '))) 
    valids = 0
    for i in range(len(results.index)):        
        result = results.loc[i]        
        cid = result[0]
        rid = result[1]
        iid = result[2]
        uid = result[3]
        #print('result', i, ' : ', cid, rid, iid, uid)
        for j in range(len(clusters)):
            cl = clusters[j]
            filtered = False
            if filters is not None:
                for filter in filters:
                    if filter[0] == cl[0]:
                        filtered = True
                        break
            if filtered:
                continue
            #print('cluster', j, ' : ',cl)
            if cid <= cl[1][1] and cid >= cl[1][0]:
                #print(cid, " in ", cl[1])
                if rid <= cl[2][1] and rid >= cl[2][0]:
                    #print(rid, " in ", cl[2])
                    if iid <= cl[3][1] and iid >= cl[3][0]:
                        #print(iid, " in ", cl[3])
                        if uid <= cl[4][1] and uid >= cl[4][0]:
                            #print(uid, " in ", cl[4])
                            valids +=1
                            break
            #else:
             #   print(cid, " not in ", cl[1])
        #print(valids)
    return valids

data_path = 'gan_data.csv'
init_stage = True
update_stage = False
gen_stage = False
active_mode = True
latent_dim = 10
input_dim = 4
n_classes = 2
n_samples = 1000000
if init_stage:
    init_model(latent_dim, input_dim,  n_classes, n_samples, active_mode, data_path)
elif update_stage:
    update_model(latent_dim, input_dim, n_classes, n_samples, active_mode, data_path)
if gen_stage:    
    report = list()
    for i in range(10, 110, 10):  
        results = DataFrame()
        while len(results.index) < i:  
            X = generate_good_samples(latent_dim, n_classes, i, 1)
            X_df = DataFrame(X.reshape((X.shape[0], X.shape[1]))) 
            if  len(results.index) == 0:
                results = X_df
            else:
                results = results.append(X_df, ignore_index=True)
            results = results.drop_duplicates(ignore_index=True)
            if len(results.index) > i:                
                results = results[:i]                  
            
        clusters = read_csv('gan_clusters.csv').values.tolist()    
          
        valids = check_validity(results, clusters)        
        report.append((i, valids))
        print(i, valids)
    output = DataFrame(report, columns=['total TCS', 'Passive TCs'])
    output.to_csv('gan_tcs_active.csv')
    print(output)

    

