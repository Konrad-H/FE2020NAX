# NEURAL NETWORK FOR ENERGY CONSUMPTION PREDICTION


# key packages
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# import custom loss
from custom_loss import custom_loss

# load the dataset


# y test and train variables are 1D power consumption
# y predicted variables are 2D, mean and variance




# x test and train variables are 10D
# 2 dimensions are for weather effects 

weather_var = ["drybulb",	"dewpnt"]
calendar_var = ["date","year",	"month",	"hour",
 	"day_of_week", 	"day_of_year",	"weekend", 	
     "holiday_name",	"holiday"]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



