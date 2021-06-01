#import libraries
import os
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten,  MaxPooling2D, Conv2D, AveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import gplearn
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


#import data
data_in=pd.read_csv('changed_input.csv')  # input csv file with columns of input variables
data_out=pd.read_csv('changed_output.csv') #output csv file with column of results
X_all=np.array(data_in)
y_all=np.array(data_out)

#import more data
cities = pd.read_csv('cali_input_.csv')
lat = cities['lat'].values
lon = cities['long'].values
area = cities['area'].values

#sidebar code
st.sidebar.markdown("## Select Data Intensity and Slope Angle")
names = cities['Region'].values
select_event = st.sidebar.selectbox(' Which city do you want to modify?', names)
str_t0 = st.sidebar.slider('Intensity(mm/hr)', 1, 100, 4)
t0 = np.log(float(str_t0))
slope_angle = st.sidebar.slider('Slope Angle(degrees)', 20, 35, 28)

#modify data using sidebar
city_index = names.tolist().index(select_event)
X_all[440+city_index, 0] = t0
X_all[440+city_index, 1] = slope_angle

#model code
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
# normalize the data attributes
X_all = scaler_X.fit_transform(X_all)
X=np.array(X_all[1:353,:]) #specify portion of data to use for training
y_all = scaler_y.fit_transform(y_all)
y=np.array(y_all[1:353,:]) #repeat here
#X_all.shape

model=Sequential([
    Dense(5, input_dim=2,activation='relu'), #change input_dim to however many input variables
    Dense(5, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='relu')
])

model.summary()
model.compile(Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])
history=model.fit(X,y, validation_split=0.1, batch_size=10, epochs=100, shuffle=True, verbose=2)
test_x=X_all[440:456,:] # specify testing dataset outside of training dataset
print(test_x)
predictions=model.predict(test_x, batch_size=10, verbose=0)
y_pred=scaler_y.inverse_transform(predictions)

# model.save('Cure.h5')
# model.save_weights('Cure.h5')
# x_data=test_x
# x_data=scaler_X.inverse_transform(x_data)

#title and graphs
st.title('Rainfall-induced Landslide simulator')
st.markdown("""
 * Use the menu to the left to modify the data
 * Your plots will appear below
""")

failure = [10**x for x in y_pred]
data = failure
col1, col2 = st.beta_columns(2)
col1.dataframe(failure)
col2.dataframe(names)
st.line_chart(data)
