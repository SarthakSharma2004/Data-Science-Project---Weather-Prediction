#!/usr/bin/env python
# coding: utf-8

# # INSTALLING ALL THE LIBRARIES

# In[1]:


get_ipython().system('pip3 install neuralprophet')


# In[2]:


import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle



# # IMPORTING THE DATA AND PREPROCESSING IT

# In[3]:


df = pd.read_csv("weatherAUS.csv")
df.head() #for first 5 rows of the dataset


# In[4]:


df.Location.unique() #all the unique values within a column


# In[5]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.isnull.sum()


# In[11]:


df.duplicated().sum()


# # We Will Forecast Weather Of Melbourne
# 

# In[12]:


melb = df[df['Location']=='Melbourne'] #specifying the location to melbourne
melb['Date'] = pd.to_datetime(melb['Date']) #To change the 'date' data tyype of Melboune to 'datetime' datatype
melb.head()


# In[13]:


melb.dtypes.head()


# In[14]:


#PLOTTING GRAPH

plt.figure(figsize=(10, 6))  # Set the figure size

# Plot the data with specific styling
plt.plot(melb['Date'], melb['Temp3pm'], color='skyblue', linewidth=2)

# Add labels and title
plt.title('Melbourne Temperature at 3 PM Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Add a background color for better contrast
plt.gca().set_facecolor('#f9f9f9')

# Show plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[15]:


melb.tail()


# In[16]:


#Keeping only the Date and Temp (at 3PM) columns
data = melb[['Date','Temp3pm']]
data.dropna(inplace = True)
data.columns = ['ds', 'y']
data.head()


# # Training Our Model

# In[17]:


m = NeuralProphet()
m.fit(data, freq = 'D', epochs = 800) #Puting Data into the model, set the frequency to Daily ('D')


# # Let's Forecast
# 

# In[18]:


future = m.make_future_dataframe(data, periods = 3300) #forecasting for 900 periods
forecast = m.predict(future)
forecast.head()


# In[20]:


forecast.tail()
#yjat1 represents the actual forecast value


# In[21]:


get_ipython().system('pip3 install plotly-resampler')


# In[22]:


m.plot(forecast)


# In[23]:


m.plot_components(forecast)


# # Saving The Model

# In[26]:


with open("forecast_model.pkl",'wb') as f:
    pickle.dump(m,f)
    


# In[27]:


m


# In[30]:


with open('forecast_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    


# In[31]:


m


# In[ ]:




