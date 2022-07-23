import keras
from keras import layers
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras import regularizers
from sklearn import preprocessing
import numpy as np
import sklearn
import librosa
import librosa.display
from scipy import signal
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import ast
from google.colab import drive
drive.mount('/content/drive')



def segmentation(x,segLength):
    Segs=[]
    Segs1=[]
    Segs2=[]
    num=x.shape[0]//segLength
    for i in range(0,num):
        segx=x[i*segLength:(i+1)*segLength]
        Segs1.append(segx)
    num=(x.shape[0]-segLength//2)//segLength
    for i in range(0,num):
        segx=x[i*segLength+segLength//2:i*segLength+segLength//2+segLength]
        Segs2.append(segx)
    for i in range(0,min(len(Segs1),len(Segs2))):
        Segs.append(Segs1[i])
        Segs.append(Segs2[i])
    if len(Segs1)>len(Segs2):
        Segs.append(Segs1[len(Segs1)-1])
    if len(Segs1)<len(Segs2):
        Segs.append(Segs2[len(Segs2)-1])
    return Segs

def plotSpectrogram(x,smpRate,hopSize,Yaxis,figName,fmat,tmat):
    
    plt.figure(figsize=(15,9))
    librosa.display.specshow(x,sr=smpRate,hop_length=hopSize,x_axis='time',y_axis=Yaxis,vmin=-0.5,vmax=0.5,cmap='bwr')
    plt.colorbar(format='%+2.f')
    
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(x, sr=smpRate, ax=ax,
                                 hop_length=hopSize, y_axis='linear', x_axis='time',vmin=-0.5,vmax=0.5,cmap='bwr')
    fig.savefig(figName)
    
    
    

def getSpecMatrix(x,smpRate):
    frameSize=256
    hopSize=128
    S_scale=librosa.stft(x, n_fft=frameSize, hop_length=hopSize, )
    Y_scale=np.abs(S_scale)**2
    Y_log_scale=librosa.power_to_db(Y_scale)
    return Y_log_scale

def getFeatureMat(audio_file,smpRate):
    data,fr=librosa.load(audio_file,sr=smpRate)
    dataMat=segmentation(data)
    FeatureMat=[]
    for line in dataMat:
        mat=getSpecMatrix(line,fr)
        FeatureMat.append(mat)
    return np.array(FeatureMat)

def getPcaMat(dataMat,nComponents):
    pca=PCA(n_components=nComponents, whiten=True)
    nDataMat=[]
    for mat in dataMat:
        nmat=mat.flatten()
        nDataMat.append(nmat)
    nDataMat=np.array(nDataMat)
    trans_dataMat=pca.fit_transform(nDataMat)
    trans_dataMat=np.array(trans_dataMat)
    return trans_dataMat

def loadDataBatch(fileLocation):
    f=open(fileLocation)
    inData=f.read()
    arrFile=inData.split('\n')
    dataBatch=[]
    for fileName in arrFile:
        if fileName=='':
            continue
        fileName="drive/MyDrive/Colab Notebooks/data/"+fileName
        x,fr=librosa.load(fileName, sr=16000)
        chunks=segmentation(x,3456)
        dataBatch+=chunks
    dataBatch=np.array(dataBatch)
    sDataBatch=[]
    for line in dataBatch:
        SpecData=getSpecMatrix(line, 16000)
        #print(SpecData.shape)
        SpecLine=SpecData.flatten()
        sDataBatch.append(SpecLine)
    sDataBatch=np.array(sDataBatch)
    return sDataBatch

def De_Flatten(OrgMat,nrow,ncol):
    resMat=[]
    for i in range(0,nrow):
        tmp=OrgMat[i*ncol:i*ncol+ncol]
        resMat.append(tmp)
    resMat=np.array(resMat)
    return resMat


data_timit=loadDataBatch('drive/MyDrive/Colab Notebooks/data/fileName_TIMIT.txt')
All_data=data_timit
All_data=np.array(All_data)
data_eng=loadDataBatch('drive/MyDrive/Colab Notebooks/data/fileName_Eng.txt')
data_dutch=loadDataBatch('drive/MyDrive/Colab Notebooks/data/fileName_Dutch.txt')
data_french=loadDataBatch('drive/MyDrive/Colab Notebooks/data/fileName_French.txt')
data_german=loadDataBatch('drive/MyDrive/Colab Notebooks/data/fileName_German.txt')
data_port=loadDataBatch('drive/MyDrive/Colab Notebooks/data/fileName_Port.txt')
data_arab=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-ARAB/fileName.txt')
data_bulga=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-BULGA/fileName.txt')
data_canto=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-CANTO/fileName.txt')
data_catalan=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-CATALAN/fileName.txt')
data_hebrew=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-HEBREW/fileName.txt')
data_hindi=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-HINDI/fileName.txt')
data_japan=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-JAPAN/fileName.txt')
data_persian=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-PERSIAN/fileName.txt')
data_sindhi=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-SINDHI/fileName.txt')
data_swe=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-SWE/fileName.txt')
data_turk=loadDataBatch('drive/MyDrive/Colab Notebooks/data/ALL-TURK/fileName.txt')
#print(data_timit.shape,data_eng.shape,data_dutch.shape,data_french.shape,data_german.shape,data_port.shape)

All_data=np.concatenate([All_data,data_eng],axis=0)
All_data=np.concatenate([All_data,data_dutch],axis=0)
All_data=np.concatenate([All_data,data_french],axis=0)
All_data=np.concatenate([All_data,data_german],axis=0)
All_data=np.concatenate([All_data,data_port],axis=0)
All_data=np.concatenate([All_data,data_arab],axis=0)
All_data=np.concatenate([All_data,data_bulga],axis=0)
All_data=np.concatenate([All_data,data_canto],axis=0)
All_data=np.concatenate([All_data,data_catalan],axis=0)
All_data=np.concatenate([All_data,data_hebrew],axis=0)
All_data=np.concatenate([All_data,data_hindi],axis=0)
All_data=np.concatenate([All_data,data_japan],axis=0)
All_data=np.concatenate([All_data,data_persian],axis=0)
All_data=np.concatenate([All_data,data_sindhi],axis=0)
All_data=np.concatenate([All_data,data_swe],axis=0)
All_data=np.concatenate([All_data,data_turk],axis=0)

print(All_data.shape)
print('All Data:')
print(All_data.mean(),All_data.max(),All_data.min())

pca=PCA(n_components=200, whiten=True)
nDataMat=[]
for mat in All_data:
    nmat=mat.flatten()
    nDataMat.append(nmat)
nDataMat=np.array(nDataMat)
trans_dataMat=pca.fit_transform(nDataMat)
trans_dataMat=np.array(trans_dataMat)
pca_all=trans_dataMat
print('pca:')
print(sum(pca.explained_variance_ratio_))
print(pca_all.shape)
print('pca_all')
print(pca_all.mean(),pca_all.max(),pca_all.min())
template_pca_all=pca_all
pca_all=(pca_all-pca_all.min())/(pca_all.max()-pca_all.min())
print(pca_all.mean(),pca_all.max(),pca_all.min())

encoding_dim = 400  

input_img = keras.Input(shape=(200,))

h_layer=layers.Dense(encoding_dim, activation='tanh',activity_regularizer=regularizers.l1(0.001))
encoded = h_layer(input_img)
out_layer=layers.Dense(200, activation='sigmoid')
decoded = out_layer(encoded)


autoencoder = keras.Model(input_img, decoded)



encoder = keras.Model(input_img, encoded)



encoded_input = keras.Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = keras.Model(encoded_input, decoder_layer(encoded_input))



autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])




test_index=random.sample(range(0,pca_all.shape[0]-1),1500)
x_train=pca_all
x_train=np.array(x_train)
x_test=[]
for test_index_item in test_index:
    x_test.append(x_train[test_index_item])
x_test=np.array(x_test)
x_train=np.delete(x_train,test_index,axis=0)
print(x_train.shape)
print(x_test.shape)




history=autoencoder.fit(x_train, x_train,
                epochs=700,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


wmat=out_layer.get_weights()
dict_mat=wmat[0]
print('dict_mat:')
print(dict_mat.mean(),dict_mat.max(),dict_mat.min())
dict_mat=(dict_mat*(template_pca_all.max()-template_pca_all.min()))+template_pca_all.min()
error_mat=decoded_imgs-x_test
print(error_mat.mean(),error_mat.max(),error_mat.min())
decoded_imgs=(decoded_imgs*(template_pca_all.max()-template_pca_all.min()))+template_pca_all.min()
decoded_imgs_org=pca.inverse_transform(decoded_imgs)
print(decoded_imgs_org)
print(decoded_imgs_org.mean(),decoded_imgs_org.max(),decoded_imgs_org.min())
print('decoded:',decoded_imgs.shape)

org_dict_mat=pca.inverse_transform(dict_mat)
org_dict_mat=(org_dict_mat-org_dict_mat.mean())/(org_dict_mat.max()-org_dict_mat.min())


x_test_denorm=(x_test*(template_pca_all.max()-template_pca_all.min()))+template_pca_all.min()
x_test_org=pca.inverse_transform(x_test_denorm)

for i in range(0,org_dict_mat.shape[0]):
  for j in range(0,org_dict_mat.shape[1]):
    if abs(org_dict_mat[i][j])<=0.05:
      org_dict_mat[i][j]=0

for i in range(0,400):
    spc_data=De_Flatten(org_dict_mat[i],129,28)
    st='drive/MyDrive/Colab Notebooks/data/human_400/'+str(i)+'.png'
    plotSpectrogram(spc_data,16000,128,'log',st,f_stft,t_stft)
    





