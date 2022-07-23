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
from imblearn.over_sampling import SMOTE
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



def loadData(fileLocation,smpRate):
    f=open(fileLocation)
    inData=f.read()
    arrFile=inData.split('\n')
    audioFiles=[]
    for fileName in arrFile:
        if fileName=='':
            continue
        fileName="drive/MyDrive/Colab Notebooks/data/"+fileName
        x,fr=librosa.load(fileName, sr=smpRate)
        audioFiles.append(x)
    return audioFiles

def getSpecMatrix(x,smpRate):
    frameSize=256
    hopSize=128
    S_scale=librosa.stft(x, n_fft=frameSize, hop_length=hopSize, )
    #print(S_scale)
    Y_scale=np.abs(S_scale)**2
    Y_log_scale=librosa.power_to_db(Y_scale)
    return Y_log_scale

def plotSpectrogram(x,smpRate,hopSize,Yaxis,figName):
    
    plt.figure(figsize=(15,9))
    librosa.display.specshow(x,sr=smpRate,hop_length=hopSize,x_axis='time',y_axis=Yaxis,vmin=-0.4,vmax=0.4,cmap='bwr')
    plt.colorbar(format='%+2.f')
    
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(x, sr=smpRate, ax=ax,
                                 hop_length=hopSize, y_axis='linear', x_axis='time',vmin=-0.4,vmax=0.4,cmap='bwr')
    fig.savefig(figName)

def De_Flatten(OrgMat,nrow,ncol):
    resMat=[]
    for i in range(0,nrow):
        tmp=OrgMat[i*ncol:i*ncol+ncol]
        resMat.append(tmp)
    resMat=np.array(resMat)
    return resMat

All_org_data=loadData('drive/MyDrive/Colab Notebooks/data/3 types bat calls/FM/fileName.txt',250000)
All_segs=[]
for line in All_org_data:
    chunks=segmentation(line,3125)
    All_segs+=chunks

All_segs=np.array(All_segs)
print('orgiginal data shape:')
print(All_segs.shape)

dataPlaceholder=[]
nExpectedSample=All_segs.shape[0]*20
delSegList=[]

for i in range(0,All_segs.shape[0]):
    if sum(All_segs[i])<0.01 and max(All_segs[i])<0.001:
        delSegList.append(i)

All_segs_del=np.delete(All_segs,delSegList,0)

for i in range(0,nExpectedSample):
    tmparr=np.zeros(3125)
    dataPlaceholder.append(tmparr)

dataPlaceholder=np.array(dataPlaceholder)

X=np.concatenate([dataPlaceholder,All_segs_del],axis=0)

y=[]
for i in range(0,nExpectedSample):
    y.append(1)
for i in range(0,All_segs_del.shape[0]):
    y.append(2)

smo = SMOTE(random_state=42)
X = X.astype('float64')
X_smo, y_smo = smo.fit_resample(X, y)
print(X_smo.shape)

delSegList=[]

for i in range(0,X_smo.shape[0]):
    if sum(X_smo[i])<0.01 and max(X_smo[i])<0.001:
        delSegList.append(i)

rsmpData=np.delete(X_smo,delSegList,0)

print('resampled data shape:')
print(rsmpData.shape)

All_spec_data=[]

for line in rsmpData:
    specData=getSpecMatrix(line, 250000)
    specData=specData.flatten()
    All_spec_data.append(specData)
All_data=np.array(All_spec_data)

pca=PCA(n_components=200, whiten=True)
nDataMat=[]
for mat in All_data:
    nmat=mat.flatten()
    nDataMat.append(nmat)
nDataMat=np.array(nDataMat)
trans_dataMat=pca.fit_transform(nDataMat)
trans_dataMat=np.array(trans_dataMat)
pca_all=trans_dataMat
print('shape after pca')
print(pca_all.shape)
print('pca_all')
print(pca_all.mean(),pca_all.max(),pca_all.min())
template_pca_all=pca_all
pca_all=(pca_all-pca_all.min())/(pca_all.max()-pca_all.min())
print('normalized pca')
print(pca_all.mean(),pca_all.max(),pca_all.min())

print('pca explained variance')
print(sum(pca.explained_variance_ratio_))


encoding_dim = 400  

input_img = keras.Input(shape=(200,))

h_layer=layers.Dense(encoding_dim, activation='tanh', activity_regularizer=regularizers.l1(0.005))
encoded = h_layer(input_img)
out_layer=layers.Dense(200, activation='sigmoid')
decoded = out_layer(encoded)


autoencoder = keras.Model(input_img, decoded)



encoder = keras.Model(input_img, encoded)



encoded_input = keras.Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])






test_index=random.sample(range(0,pca_all.shape[0]-1),4000)
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
                epochs=2000,
                batch_size=512,
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
x_test_denorm=(x_test*(template_pca_all.max()-template_pca_all.min()))+template_pca_all.min()
x_test_org=pca.inverse_transform(x_test_denorm)
decoded_imgs_org=pca.inverse_transform(decoded_imgs)
print(decoded_imgs_org)
print(decoded_imgs_org.mean(),decoded_imgs_org.max(),decoded_imgs_org.min())
print('decoded:',decoded_imgs.shape)

org_dict_mat=pca.inverse_transform(dict_mat)
org_dict_mat=(org_dict_mat-org_dict_mat.mean())/(org_dict_mat.max()-org_dict_mat.min())




x_test_denorm=(x_test*(template_pca_all.max()-template_pca_all.min()))+template_pca_all.min()
x_test_org=pca.inverse_transform(x_test_denorm)

#thresholding

for i in range(0,org_dict_mat.shape[0]):
  for j in range(0,org_dict_mat.shape[1]):
    if abs(org_dict_mat[i][j])<=0.05:
      org_dict_mat[i][j]=0

for i in range(0,400):
    spc_data=De_Flatten(org_dict_mat[i],129,25)
    st='drive/MyDrive/Colab Notebooks/data/bats_400/'+str(i)+'.png'
    plotSpectrogram(spc_data,250000,128,'linear',st)
