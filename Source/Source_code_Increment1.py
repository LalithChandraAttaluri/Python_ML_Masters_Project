#!/usr/bin/env python
# coding: utf-8

# In[63]:


#Importing Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


#Converting our multiple CSV files to dataframes
crimes_2005_2007=pd.read_csv('Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)
crimes_2001_2004=pd.read_csv('Chicago_Crimes_2001_to_2004.csv',error_bad_lines=False)
crimes_2008_2011=pd.read_csv('Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)
crimes_2012_2017=pd.read_csv('Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)


# In[65]:


#Sampling each data frame to 50K samples
crimes_2001_2004 = crimes_2001_2004.sample(n=50000)
crimes_2005_2007 = crimes_2005_2007.sample(n=50000)
crimes_2008_2011 = crimes_2008_2011.sample(n=50000)
crimes_2012_2017 = crimes_2012_2017.sample(n=50000)


# In[66]:


crimes_2001_2004.shape


# In[67]:


#Concatenating all the dataframes to a single dataframe
data_frames=[crimes_2001_2004, crimes_2005_2007, crimes_2008_2011, crimes_2012_2017]
crimes=pd.concat(data_frames)


# In[68]:


crimes.shape


# In[69]:


crimes.head


# In[70]:


#Removing Duplicate Records
print('Dataset ready..')
print('Dataset Shape before drop_duplicate : ', crimes.shape)
crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print('Dataset Shape after drop_duplicate: ', crimes.shape)


# In[71]:


# convert dates to pandas datetime format
crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date will help us a lot later on
crimes.index = pd.DatetimeIndex(crimes.Date)


# In[72]:


crimes.info()


# In[73]:


#Transforming the least used cateogiries to a single cateogory "OTHER"
loc_to_change  = list(crimes['Location Description'].value_counts()[20:].index)
print(loc_to_change)


# In[74]:


crimes.loc[crimes['Location Description'].isin(loc_to_change) , crimes.columns=='Location Description'] = 'OTHER'


# In[75]:


crimes['Location Description']


# In[76]:


desc_to_change = list(crimes['Description'].value_counts()[20:].index)
print(desc_to_change)


# In[77]:


crimes.loc[crimes['Description'].isin(desc_to_change) , crimes.columns=='Description'] = 'OTHER'
crimes['Description']


# In[78]:


#Calaculationg percentage of Null values
percent_missing = crimes.isnull().sum()/ len(crimes) * 100
percent_missing


# In[79]:


#Dropping Null values
crimes = crimes.dropna()
crimes.isnull().sum()


# In[80]:


crimes.head()
crimes[['District', 'Ward','Community Area']] = crimes[['District', 'Ward','Community Area']].astype('int')
crimes[['District', 'Ward','Community Area']] = crimes[['District', 'Ward','Community Area']].astype('str')
crimes.head()


# In[81]:


crimes.columns = crimes.columns.str.strip().str.lower().str.replace(' ', '_')
crimes.head(5)


# In[82]:


#Plotting geographical map of crimes commited with respect to wards
#definition of the boundaries in the map
district_geo = r'Boundaries-Wards.geojson'
Chicago_COORDINATES = (41.895140898, -87.624255632)
import folium
#calculating total number of incidents per district for 2016
WardData2016 = pd.DataFrame(crimes['ward'].value_counts().astype(float))
WardData2016.to_json('Ward_Map.json')
WardData2016 = WardData2016.reset_index()
WardData2016.columns = ['ward', 'Crime_Count']
 
#creating choropleth map for Chicago District 2016
map1 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)
map1.choropleth(geo_data = district_geo, 
                #data_out = 'Ward_Map.json', 
                data = WardData2016,
                columns = ['ward', 'Crime_Count'],
                key_on = 'feature.properties.ward',
                fill_color = 'YlOrRd', 
                fill_opacity = 0.7, 
                line_opacity = 0.2)
map1


# In[26]:


#Plotting geographicl map of Crimes with respect to Districts
#definition of the boundaries in the map
district_geo = r'Boundaries_Police_Districts.geojson'

district_data = pd.DataFrame(crimes['district'].value_counts().astype(float))
district_data.to_json('District_Map.json')
district_data = district_data.reset_index()
district_data.columns = ['district', 'Crime_Count']

#creation of the choropleth
map2 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)
map2.choropleth(geo_data = district_geo,  
                data = district_data,
                columns = ['district', 'Crime_Count'],
                key_on = "feature.properties.dist_num",
                fill_color = 'YlOrRd', 
                fill_opacity = 0.7, 
                line_opacity = 0.2)
map2


# In[83]:


# Splitting the Date to Day, Month, Year, Hour, Minute, Second
crimes['date2'] = pd.to_datetime(crimes['date'])
crimes['Year'] = crimes['date2'].dt.year
crimes['Month'] = crimes['date2'].dt.month
crimes['Day'] = crimes['date2'].dt.day
crimes['Hour'] = crimes['date2'].dt.hour
crimes['Minute'] = crimes['date2'].dt.minute
crimes['Second'] = crimes['date2'].dt.second 
crimes = crimes.drop(['date'], axis=1) 
crimes = crimes.drop(['date2'], axis=1) 
crimes.head()


# In[84]:


crimes.info()


# In[85]:


#Plotting monthly crimes commited every year
crimes.groupby(['Month','Year'])['id'].count().unstack().plot(marker='o', figsize=(15,10))
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.xticks(np.arange(12),months)
plt.ylabel('No of Crimes')

plt.show()


# In[86]:


#Crimes with respect to cateogories
plt.figure(figsize=(14,10))
plt.title('Amount of Crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Amount of Crimes')
crimes.groupby([crimes['primary_type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.show()


# In[87]:


all_classes = crimes.groupby(['primary_type'])['block'].size().reset_index()
all_classes


# In[88]:


all_classes['amt'] = all_classes['block']
all_classes = all_classes.drop(['block'], axis=1)
all_classes


# In[89]:


all_classes = all_classes.sort_values(['amt'], ascending=[False])
unwanted_classes = all_classes.tail(12)
unwanted_classes


# In[90]:


crimes.loc[crimes['primary_type'].isin(unwanted_classes['primary_type']), 'primary_type'] = 'OTHERS'
plt.figure(figsize=(14,10))
plt.title('Amount of Crimes by Primary Type')
plt.ylabel('Crime Type')
plt.xlabel('Amount of Crimes')
crimes.groupby([crimes['primary_type']]).size().sort_values(ascending=True).plot(kind='barh', stacked=True, colormap = 'Paired_r')
plt.show()


# In[94]:


a=crimes['primary_type'].unique()
print(a)


# In[95]:



#Encoding the object type features
crimes['block'] = pd.factorize(crimes["block"])[0]
crimes['primary_type'] = pd.factorize(crimes["primary_type"])[0]
crimes['description'] = pd.factorize(crimes["description"])[0]
crimes['location_description'] = pd.factorize(crimes["location_description"])[0]
crimes['district'] = pd.factorize(crimes["district"])[0]
crimes['ward'] = pd.factorize(crimes["ward"])[0]
crimes['community_area'] = pd.factorize(crimes["community_area"])[0]


# In[96]:


#Finding Correlation
x = crimes.drop(['primary_type'], axis=1)
y = crimes['primary_type']
import seaborn as sns; sns.set(color_codes=True)
plt.figure(figsize=(20,12))
correlation = crimes.corr()
sns.heatmap(correlation, annot=True, cmap='viridis')
plt.show()


# In[97]:


#Finding Correlation with Target
correlation_target=abs(correlation['primary_type'])
print(correlation_target)


# In[98]:


#Selecting highly correlated features
relevant_features = correlation_target[correlation_target>0.1]
relevant_features


# In[99]:


features=["description", "arrest"]
print('The features that are more correlated to the model are: ', features)


# In[100]:


#Splitting data into train and test
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(crimes ,test_size=0.4, random_state=0)


# In[101]:


target='primary_type'
x_train=train_data[features]
y_train=train_data[target]
x_test=test_data[features]
y_test=test_data[target]


# In[102]:


#K-nearest neighbours Model
from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=4)
model_knn.fit(x_train,y_train)


# In[103]:


#Predicting the result
predicted_result=model_knn.predict(x_test)


# In[104]:


# Evaluating the model
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
accuracy = accuracy_score(y_test, predicted_result)
recall = recall_score(y_test, predicted_result, average="weighted")
percision = precision_score(y_test, predicted_result, average="weighted")
f1score = f1_score(y_test, predicted_result, average='micro')
confusionmatrix = confusion_matrix(y_test, predicted_result)

print("========== Evaluation results of KNN Model ==========")
print("Accuracy    : ", accuracy)
print("Recall      : ", recall)
print("Precision   : ", percision)
print("F1 Score    : ", f1score)
print("Confusion Matrix: ")
print(confusionmatrix)


# In[105]:


target_names = crimes['primary_type'].unique()
print(target_names)


# In[115]:


# Classification Report
# Instantiate the classification model and visualizer
from yellowbrick.classifier import ClassificationReport
target_names = a
visualizer = ClassificationReport(model_knn, classes=target_names, size=(1080, 720))
visualizer.fit(X=x_train, y=y_train)     # Fit the training data to the visualizer
print(visualizer.score(x_test, y_test))      # Evaluate the model on the test data
print(classification_report(y_test, predicted_result,target_names=a))
g = visualizer.poof()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




