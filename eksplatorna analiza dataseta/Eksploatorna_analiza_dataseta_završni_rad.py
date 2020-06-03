import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from pandas.plotting import table
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn import tree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dataset=pd.read_csv("heart.csv")

###############################################Statistička analiza podataka#########################


buffer = io.StringIO()
dataset.info(buf=buffer)
data1 = buffer.getvalue()
with open("df_info.txt", "w",
          encoding="utf-8") as f:
    f.write(data1)
desc=dataset[['age','sex','cp']].describe()
desc1=dataset[['trestbps','chol','fbs']].describe()
desc2=dataset[['restecg','thalach','exang']].describe()
desc3=dataset[['oldpeak','slope','ca','thal']].describe()
plt.figure(figsize=(20,10))
plot=plt.subplot(221,frame_on=False)
plot1=plt.subplot(222,frame_on=False)
plot2=plt.subplot(224,frame_on=False)
plot3=plt.subplot(223,frame_on=False)
plot.xaxis.set_visible(False)
plot.yaxis.set_visible(False)
plot1.xaxis.set_visible(False)
plot1.yaxis.set_visible(False)
plot2.xaxis.set_visible(False)
plot2.yaxis.set_visible(False)
plot3.xaxis.set_visible(False)
plot3.yaxis.set_visible(False)
table(plot,desc,loc='center',fontsize=10)
table(plot1,desc1,loc='center',fontsize=10)
table(plot2,desc2,loc='center',fontsize=10)
table(plot3,desc3,loc='center',fontsize=10)
plt.savefig('dataset_description.png')
plt.close()


#Crtanje matrice koleracije za podatke





plt.figure(figsize=(10,10))
plt.matshow(dataset.corr(), fignum=1)
plt.yticks(np.arange(dataset.shape[1]), dataset.columns,fontsize=20)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns,rotation=90,fontsize=20)
plt.colorbar().ax.tick_params(labelsize=20)

plt.savefig("korelacija.png")

plt.close()
#Crtanje histograma

plt.style.use(['classic','ggplot'])
dataset.hist(xlabelsize=7,ylabelsize=8,figsize=(10,10))

plt.savefig("histogram.png")
plt.close()

#Predprocesiranje podataka
x=dataset.drop(['target'],axis=1)
y=dataset['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_test_pocetni=x_test
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
classificator_predictions=[]

##########################################STROJNO UCENJE#############################

#Model Support vector machine

klasifikator=SVC(kernel='linear')
klasifikator.fit(x_train,y_train)
y_pred_test=klasifikator.predict(x_test)
confusion_matrix_test=confusion_matrix(y_pred_test,y_test)
y_pred_trained=klasifikator.predict(x_train)
confusion_matrix_train=confusion_matrix(y_pred_trained,y_train)
print_matrix=sns.heatmap(confusion_matrix_test,annot=True,xticklabels=['Pozitivan','Negativan'],yticklabels=['Pozitivan','Negativan'],cbar=False,fmt='d',annot_kws={"size": 30})

print_matrix.tick_params(labelsize=25)
print('Predikcija za trening set nad modelom support vector machine je {}'.format((confusion_matrix_train[0][0]+confusion_matrix_train[1][1])/len(y_train)))
print('Predikcija za test set nad modelom support vector machine je {}'.format((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)))
print()
classificator_predictions.append((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)*100)
plt.title('Confusion matrix for test data',fontsize=25,color='r')
plt.savefig('confusion_matrix_test_SVM')
plt.close()
print_matrix=sns.heatmap(confusion_matrix_train,annot=True,xticklabels=['Positive','Negative'],yticklabels=['Positive','Negative'],cbar=False,fmt='d',annot_kws={"size": 30})
print_matrix.tick_params(labelsize=25)
plt.title('Confusion matrix for train data',fontsize=25,color='r')
plt.savefig('confusion_matrix_train_SVM')
plt.close()
print(classification_report(y_test,y_pred_test,target_names=['0','1']))

#Model Naive Bayes-GaussianNB klasifikator

klasifikator=GaussianNB()
klasifikator.fit(x_train,y_train)
y_pred_test=klasifikator.predict(x_test)
y_pred_train=klasifikator.predict(x_train)
confusion_matrix_train=confusion_matrix(y_pred_train,y_train)
confusion_matrix_test=confusion_matrix(y_pred_test,y_test)
print_matrix=sns.heatmap(confusion_matrix_test,annot=True,xticklabels=['Pozitivni','Negativni'],yticklabels=['Pozitivni','Negativni'],cbar=False,fmt='d',annot_kws={"size": 30})

print_matrix.tick_params(labelsize=25)
print('Predikcija za trening set nad modelom naive Bayes je {}'.format((confusion_matrix_train[0][0]+confusion_matrix_train[1][1])/len(y_train)))
print('Predikcija za test set nad modelom naive Bayes je {}'.format((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)))
print()
classificator_predictions.append((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)*100)
plt.title('Confusion matrix for test data',fontsize=25,color='r')
plt.savefig('confusion_matrix_test_Bayes')
plt.close()
print_matrix=sns.heatmap(confusion_matrix_train,annot=True,xticklabels=['Positive','Negative'],yticklabels=['Positive','Negative'],cbar=False,fmt='d',annot_kws={"size": 30})
print_matrix.tick_params(labelsize=25)
plt.title('Confusion matrix for train data',fontsize=25,color='r')
plt.savefig('confusion_matrix_train_Bayes')
plt.close()
print(classification_report(y_test,y_pred_test,target_names=['0','1']))

#Logistic regresion

klasifikator=LogisticRegression()
klasifikator.fit(x_train,y_train)
y_pred_test=klasifikator.predict(x_test)
y_pred_train=klasifikator.predict(x_train)
confusion_matrix_train=confusion_matrix(y_pred_train,y_train)
confusion_matrix_test=confusion_matrix(y_pred_test,y_test)
print_matrix=sns.heatmap(confusion_matrix_test,annot=True,xticklabels=['Pozitivni','Negativni'],yticklabels=['Pozitivni','Negativni'],cbar=False,fmt='d',annot_kws={"size": 30})

print_matrix.tick_params(labelsize=25)
print('Predikcija za trening set nad modelom logistic regresion je {}'.format((confusion_matrix_train[0][0]+confusion_matrix_train[1][1])/len(y_train)))
print('Predikcija za test set nad modelom logistic regresion je {}'.format((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)))
print()
classificator_predictions.append((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)*100)
plt.title('Confusion matrix for test data',fontsize=25,color='r')
plt.savefig('confusion_matrix_test_logreg')
plt.close()
print_matrix=sns.heatmap(confusion_matrix_train,annot=True,xticklabels=['Pozitivni','Negativni'],yticklabels=['Pozitivni','Negativni'],cbar=False,fmt='d',annot_kws={"size": 30})
print_matrix.tick_params(labelsize=25)
plt.title('Confusion matrix for train data',fontsize=25,color='r')
plt.savefig('confusion_matrix_train_logreg')
plt.close()
print(classification_report(y_test,y_pred_test,target_names=['0','1']))

#Decision trees

klasifikator=DecisionTreeClassifier()
klasifikator.fit(x_train,y_train)
y_pred_test=klasifikator.predict(x_test)
y_pred_train=klasifikator.predict(x_train)
confusion_matrix_train=confusion_matrix(y_pred_train,y_train)
confusion_matrix_test=confusion_matrix(y_pred_test,y_test)
print_matrix=sns.heatmap(confusion_matrix_test,annot=True,xticklabels=['Pozitivni','Negativni'],yticklabels=['Pozitivni','Negativni'],cbar=False,fmt='d',annot_kws={"size": 30})

print_matrix.tick_params(labelsize=25)
print('Predikcija za trening set nad modelom decision tree je {}'.format((confusion_matrix_train[0][0]+confusion_matrix_train[1][1])/len(y_train)))
print('Predikcija za test set nad modelom decision tree je {}'.format((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)))
print()
classificator_predictions.append((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)*100)
plt.title('Confusion matrix for test data',fontsize=25,color='r')
plt.savefig('confusion_matrix_test_tree')
plt.close()

print_matrix=sns.heatmap(confusion_matrix_train,annot=True,xticklabels=['Positive','Negative'],yticklabels=['Positive','Negative'],cbar=False,fmt='d',annot_kws={"size": 30})
print_matrix.tick_params(labelsize=25)
plt.title('Confusion matrix for train data',fontsize=25,color='r')
plt.savefig('confusion_matrix_train_tree')
plt.close()
print(classification_report(y_test,y_pred_test,target_names=['0','1']))

#Random forest

klasifikator=RandomForestClassifier(n_estimators=2)
klasifikator.fit(x_train,y_train)
y_pred_test=klasifikator.predict(x_test)
y_pred_train=klasifikator.predict(x_train)
confusion_matrix_train=confusion_matrix(y_pred_train,y_train)
confusion_matrix_test=confusion_matrix(y_pred_test,y_test)
print_matrix=sns.heatmap(confusion_matrix_test,annot=True,xticklabels=['Pozitivni','Negativni'],yticklabels=['Pozitivni','Negativni'],cbar=False,fmt='d',annot_kws={"size": 30})

print_matrix.tick_params(labelsize=25)
print('Predikcija za trening set nad modelom random forest je {}'.format((confusion_matrix_train[0][0]+confusion_matrix_train[1][1])/len(y_train)))
print('Predikcija za test set nad modelom random forest je {}'.format((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)))
print()
classificator_predictions.append((confusion_matrix_test[0][0]+confusion_matrix_test[1][1])/len(y_test)*100)
plt.title('Confusion matrix for test data',fontsize=25,color='r')
plt.savefig('confusion_matrix_test_random')
plt.close()
print_matrix=sns.heatmap(confusion_matrix_train,annot=True,xticklabels=['Positive','Negative'],yticklabels=['Positive','Negative'],cbar=False,fmt='d',annot_kws={"size": 30})
print_matrix.tick_params(labelsize=25)
plt.title('Confusion matrix for train data',fontsize=25,color='r')
plt.savefig('confusion_matrix_train_random')
plt.close()
print(classification_report(y_test,y_pred_test,target_names=['0','1']))

#Creating keras models
model = Sequential()
model.add(Dense(12, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model1 = Sequential()
model1.add(Dense(12, input_dim=13, activation='relu'))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
history=model.fit(x_train, y_train, epochs=50, batch_size=25, verbose=1,validation_data=(x_test,y_test))

loss,accuracy=model.evaluate(x_train,y_train,verbose=1)
print("Train accuracy: %.2f " % (accuracy*100))
loss,accuracy=model.evaluate(x_test,y_test,verbose=1)
print("Test accuracy: %.2f " % (accuracy*100))
history1=model1.fit(x_train,y_train,epochs=150,batch_size=25,verbose=0,validation_data=(x_test,y_test))
history_dict=history.history
history_dict1=history1.history
loss_values=history_dict['loss']
predict_loss=history_dict['val_loss']
predict_loss1=history_dict1['val_loss']
epochs=[i for i in range(1,51)]
epochs1=[i for i in range(1,151)]
plt.plot(epochs,loss_values,'r',label='Training loss za 50 epocha')
plt.plot(epochs,predict_loss,'b',label='Prediction loss za 50 epocha')
plt.title('Training loss vs Prediction loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Training loss')
plt.close()


plt.plot(epochs,predict_loss,'b',label='Prediction loss za 50 epocha')
plt.plot(epochs1,predict_loss1,'r',label='Prediction loss za 150 epocha')
plt.title('Prediction loss for 50 and 150 epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('50 epoch vs 150 epoch')
plt.close()

#Predikcija za testni dataset

predictions = model.predict_classes(x_test)
lista=[]
lista1=[]
lista2=[]
brojac=0
for i in y_test:
    lista.append(i)
for index,row in x_test_pocetni.iterrows():
    lista1.append([row['age'],row['sex'],row['trestbps'],row['chol'],row['fbs'],row['restecg'],row['thalach'],row['exang'],row['oldpeak'],row['slope'],row['ca'],row['thal']])
for i in x_test_pocetni:
    lista2.append(i)
print("Krivo predviđeni testni primjeri:")
print(lista2)

for i in range(len(x_test_pocetni)):
    if predictions[i][0]!=lista[i]:
        print('%s result: %d (expected %d)' % (lista1[i],predictions[i][0], lista[i]))
        brojac+=1
print("Predicted with percentage:",(1-brojac/len(x_test))*100)
classificator_predictions.append((1-brojac/len(x_test))*100)

#Crtanje stupčastog dijagrama

objects = ['SVM', 'Naive-Bayes', 'Logistic Regression', 'Decision Trees', 'Random Forest','Neural Network']
fig, ax = plt.subplots(figsize=(15, 8))
ind = np.arange(len(classificator_predictions))
width=0.75
ax.barh(ind, classificator_predictions, width, color="blue")
ax.set_yticks(ind+width/2-0.4)
ax.set_yticklabels(objects, minor=False)
plt.title('Usporedba klasifikacija')
plt.xlabel('Preciznost klasifikacije')
plt.ylabel('Metode klasifikacije')

for i, v in enumerate(classificator_predictions):
    ax.text(v, i, " "+str(round(v,3)), color='red', va='center', fontweight='bold',fontsize=20)
plt.savefig('Classification comparison')
plt.close()

for i, v in enumerate(y):
    ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

################################Eksploatorna analiza###################################

#Prikaz godina i spola
sns.distplot(dataset[dataset['sex']==1]['age'],rug=False,hist=False,label='muškarci')
sns.distplot(dataset[dataset['sex']==0]['age'],rug=False,hist=False,label='žene')
plt.legend()
plt.title('Density plot za godine po spolu')
plt.savefig('Density plot of age for sex')
plt.close()

#Prikaz godina i kolesterola
sns.distplot(dataset[dataset['sex']==1]['chol'],rug=False,hist=False,label='muškarci')
sns.distplot(dataset[dataset['sex']==0]['chol'],rug=False,hist=False,label='žene')
plt.legend()
plt.title('Density plot za kolesterol po spolu')
plt.savefig('Density plot of cholesterol for sex')
plt.close()
#Prikaz godina i kolesterola
sns.distplot(dataset[dataset['sex']==1]['chol'],rug=False,hist=False,label='muškarci')
sns.distplot(dataset[dataset['sex']==0]['chol'],rug=False,hist=False,label='žene')
plt.legend()
plt.title('Density plot za kolesterol po spolu')
plt.savefig('Density plot of cholesterol for sex')
plt.close()

#Prikaz godina i tlaka
sns.distplot(dataset[dataset['sex']==1]['trestbps'],rug=False,hist=False,label='muškarci')
sns.distplot(dataset[dataset['sex']==0]['trestbps'],rug=False,hist=False,label='žene')
plt.legend()
plt.title('Density plot za tlak po spolu')
plt.savefig('Density plot of pressure for sex')
plt.close()

#Box plotovi
fig,axes=plt.subplots(nrows=2,ncols=2)
#plt.figure(figsize=(20,10))
sns.boxplot(x='chol', data=dataset, orient='v', ax=axes[0][0])
sns.boxplot(x='oldpeak', data=dataset,  orient='v', ax=axes[0][1])
sns.boxplot(x='thalach', data=dataset,  orient='v', ax=axes[1][0])
sns.boxplot(x='trestbps', data=dataset,  orient='v', ax=axes[1][1])
plt.savefig('Boxplotovi.png')
plt.close()

#Ovisnost kolesterola o krvnom tlaku
plt.rcParams['figure.figsize'] = (8,8)
sns.scatterplot(x='chol', y='trestbps', hue='sex', size=None, data=dataset)
plt.title(' Cholesterol vs Blood pressure in rest')
plt.savefig('Cholesterol vs Blood pressure')
plt.close()

#Ovisnost
plt.rcParams['figure.figsize'] = (8,8)
sns.scatterplot(x='age', y='trestbps', hue='sex', size=None, data=dataset)
plt.title(' Age vs Blood pressure in rest')
plt.savefig('Age vs Blood pressure')
plt.close()

num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
#svi scatterplotovi
plt.figure()
sns.pairplot(dataset[num])
plt.savefig('Scatterplotovi.png')
plt.close()

plt.figure()
plt.scatter(dataset[dataset['target'] == 0]['chol'], dataset[dataset['target'] == 0]['trestbps'], marker='o', c='green', label='bez srčanog udara')
plt.scatter(dataset[dataset['target'] == 1]['chol'], dataset[dataset['target'] == 1]['trestbps'], marker='x', c='red', label='srčani udar')
plt.legend()
plt.savefig("Scatterplot-target1")

plt.close()

plt.figure()
plt.scatter(dataset[dataset['target'] == 0]['age'], dataset[dataset['target'] == 0]['trestbps'], marker='o', c='green', label='bez srčanog udara')
plt.scatter(dataset[dataset['target'] == 1]['age'], dataset[dataset['target'] == 1]['trestbps'], marker='x', c='red', label='srčani udar')
plt.legend()
plt.savefig("Scatterplot-target2")

plt.close()
