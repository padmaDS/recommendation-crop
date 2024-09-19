### https://www.kaggle.com/code/srinijagottumukkala/crop-price-model/notebook
# 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('always')

data1= pd.read_csv("data\Crop_recommendation.csv")
data2=pd.read_csv("data\csv")
data=pd.read_csv("data\soil.csv")

# print(data1.info())

# print(data2.info())

# print(data1.isnull().sum())


states=data2.state.value_counts().index
# print(states)

# plt.pie(data2.state.value_counts().to_list()[:10], labels=data2.state.value_counts().index[:10], radius=1.8, autopct="%0.2f%%")
# plt.show()

values = data2.commodity.value_counts().to_list()[:10]
labels = data2.commodity.value_counts().index[:10]

plt.figure(figsize=(15,10))
sns.barplot(x=values, y=labels)
# plt.show()

print(data1['label'].unique())

# print(data1['rice'].count())
# Select only numerical columns from data1
numerical_data = data1.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
corr_matrix = numerical_data.corr()

# Plot the heatmap
sns.heatmap(corr_matrix, annot=True)
# plt.show()

# print(sns.heatmap(data1.corr(),annot=True))

# Select only numerical columns from data1
numerical_data1 = data2.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
corr_matrix1 = numerical_data1.corr()

# Plot the heatmap
sns.heatmap(corr_matrix1, annot=True)
# plt.show()

var = data1[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = data1['label']
labels = data1['label']

temps = []
model = []

Xtrain, Xtest, Ytrain, Ytest = train_test_split(var,target,test_size = 0.2,random_state =2)

NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain,Ytrain)
predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
temps.append(x)
model.append('Naive Bayes')
# print("Naive Bayes's Accuracy is: ", x)
# print(classification_report(Ytest,predicted_values))

score = cross_val_score(NaiveBayes,var,target,cv=5)
# print(score)

### saving the NaiveBias model in a pkl file

# NB_pkl_filename = 'NBClassifier.pkl'
# NB_Model_pkl = open(NB_pkl_filename, 'wb')
# pickle.dump(NaiveBayes, NB_Model_pkl)
# NB_Model_pkl.close()

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain.values,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
temps.append(x)
model.append('RF')
# print("RF's Accuracy is: ", x)

# print(classification_report(Ytest,predicted_values))

score = cross_val_score(RF,var,target,cv=5)
# print(score)


### saving the RF model in a pkl file
RF_pkl_filename = 'RandomForest.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()

plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
# print(sns.barplot(x = temps,y = model,palette='dark'))
# Plot the barplot with y as hue and legend=False
sns.barplot(x=temps, y=model, palette='dark', hue=model, dodge=False, legend=False)

# plt.show()

accuracy_models = dict(zip(model, temps))
for k, v in accuracy_models.items():
    print (k, '-->', v)

# N P K temp humidity ph rainfall
data = np.array([[74,35,40,26.49109635,80.15836264,6.980400905,242.8640342]])
prediction = RF.predict(data)
d={}
for i in data2['commodity'].values:
    for j in data2['modal_price']:
        d[i]=j
my_list=data2['commodity'].values

print(my_list)

data=pd.read_csv("data\soil.csv")

# print(data.head())
dic={}
for i in range(len(data['State'])):
    dic[(data['State'][i])]=data['Soil_type'][i]
# print(dic)

### Testing the trained model

geoloc=input("Enter the GeoLcation:")
for i in prediction:
    i=i.capitalize()
    if i in my_list:
        print(i,"is BEST CROP to grow and profit gained is RS",d[i])
print(geoloc.capitalize(),"is majorly covered by",dic[geoloc.capitalize()] ,"Soil.")