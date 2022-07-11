#!/usr/bin/env python
# coding: utf-8

# In[268]:


# importing the libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# In[269]:


# read data from excel
bike_df = pd.read_csv("SeoulBikeData.csv",encoding='cp1252')


# In[270]:


print("Info Details")
bike_df.info()


# In[271]:


bike_df.head()


# In[272]:


#checking missing values
bike_df.isna().sum()
bike_df.isnull().sum()


# In[273]:


#checking duplicate values
print('Duplicate values : ', len(bike_df[bike_df.duplicated()]))


# # EDA

# #1. Dependency of dependent variable with independent variables

# In[274]:


sb.barplot(x = 'Holiday', y = 'Rented Bike Count', data = bike_df,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'    )


# In[275]:


sb.barplot(x = 'Functioning Day', y = 'Rented Bike Count', data = bike_df,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'    )


# In[277]:


sb.barplot(x = 'Seasons', y = 'Rented Bike Count', data = bike_df,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'    )


# In[278]:


#print the plot to find the relationship between "Rented Bike Count" and "Rainfall(mm)" 
bike_df.groupby('Rainfall(mm)').mean()['Rented Bike Count'].plot()


# In[279]:


#print the plot to find the relationship between "Rented Bike Count" and "Wind speed (m/s)" 
bike_df.groupby('Wind speed (m/s)').mean()['Rented Bike Count'].plot()


# In[280]:


#print the plot to find the relationship between "Rented Bike Count" and "Temperature(°C)" 
bike_df.groupby('Temperature(°C)').mean()['Rented Bike Count'].plot()


# In[281]:


#print the plot to find the relationship between "Rented Bike Count" and "Snowfall (cm)" 
bike_df.groupby('Snowfall (cm)').mean()['Rented Bike Count'].plot()


# In[282]:


bike_df['Rainfall(mm)'].plot(kind='hist')
plt.show()


# In[283]:


bike_df['Snowfall (cm)'].plot(kind='hist')
plt.show()


# #2. Check correlation between the variables

# In[284]:


plt.figure(figsize=(8,8))
sb.set_style("whitegrid", {"axes.facecolor": ".0"})
plot_kws={"s": 1}
sb.heatmap(bike_df.corr(),
            cmap='RdYlBu',
            annot=True,
            linewidths=0.2, 
            linecolor='black').set_facecolor('green')


# Dew Point Temperature and Temperature show strong positive correlation of 0.91. 
# So , we can plan to drop the Dew Point Temperature as the 2 variable will have similar variations

# In[285]:


print('The correlation between variables: ', bike_df.corr())


# In[286]:


bike_df.plot(kind = 'scatter',x = 'Temperature(°C)', y = 'Dew point temperature(°C)', figsize = (5,5))


# In[287]:


bike_df.plot(kind = 'scatter',x = 'Humidity(%)', y = 'Dew point temperature(°C)', figsize = (5,5))


# In[288]:


bike_df.plot(kind = 'scatter',y = 'Rented Bike Count', x = 'Temperature(°C)', figsize = (5,5))


# In[289]:


bike_df.plot(kind = 'scatter',x = 'Humidity(%)', y = 'Visibility (10m)', figsize = (5,5))


# In[290]:


#bike_df.describe(include='all').T
bike_df.describe()


# # Pre Processing Data

# #1. Split Date field into Month, Date and Year
# 

# In[291]:


# Add few variables (split the date to month, year,weekday and day)
bike_df['Date']=pd.to_datetime(bike_df['Date'])
from datetime import datetime
import datetime as dt

bike_df['Year']=bike_df['Date'].dt.year
bike_df['Month']=bike_df['Date'].dt.month
bike_df['Day']=bike_df['Date'].dt.day
bike_df['DayName']=bike_df['Date'].dt.day_name()
bike_df['Weekday'] = bike_df['DayName'].apply(lambda x : 1 if x=='Saturday' or x=='Sunday' else 0 )
bike_df=bike_df.drop(columns=['Date','DayName','Year'],axis=1)


# In[292]:


bike_df.head()


# In[293]:


bike_df['Weekday'].value_counts()


# In[294]:


bike_df.info()


# #2. Create Dummy Variables

# In[295]:


#Holiday, Functioning Day and Seasons need to have dummy variables  (object DType)
bike_df = pd.get_dummies(bike_df, columns = ['Seasons',	'Holiday',	'Functioning Day'])
bike_df

#converting the categorical variables to numerical (float) types
#bike_df['Holiday'] =  bike_df['Holiday'].astype(int)
#bike_df['Functioning Day'] =  bike_df['Functioning Day'].astype(int)
#bike_df['Seasons'] =  bike_df['Seasons'].astype(int)
  
#bike_df['Holiday'] = bike_df['Holiday'].apply(lambda x : 1 if x=='Holiday' else 0 )
#bike_df['Functioning Day'] = bike_df['Functioning Day'].apply(lambda x : 1 if x=='Yes' else 0 )

# Defining all the conditions inside a function
#def condition(x):
#    if x == 'Autumn':
#        return 1
#    elif x == 'Spring':
#        return 2
#    elif x == 'Summer':
#        return 3
#    else:
#        return 4
 
# Applying the conditions
#bike_df['Seasons'] = bike_df['Seasons'].apply(condition)
#bike_df['Seasons'] = bike_df['Seasons'].apply(lambda x : 1 if x=='Autumn' elif x=='Spring' 2 elif x =='Summer' 3 else 4)
#bike_df 


# In[298]:


#sb.pairplot(bike_df_norm)


# EDA on the new features

# In[299]:


# EDA
sb.barplot(x = 'Month', y = 'Rented Bike Count', data = bike_df,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'    )


# In[300]:


sb.barplot(x = 'Weekday', y = 'Rented Bike Count', data = bike_df,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'    )


# In[301]:


sb.barplot(x = 'Hour', y = 'Rented Bike Count', data = bike_df,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'    )


# #3. Normalise the data

# In[302]:


#1 . Check the distribution plot of y variable
sb.set(rc={'figure.figsize':(10,8)})
gr = sb.distplot(bike_df['Rented Bike Count'], bins=30)
gr.axvline(bike_df['Rented Bike Count'].mean(), color='green', linestyle='dashed', linewidth=2)
gr.axvline(bike_df['Rented Bike Count'].median(), color='red', linestyle='dashed', linewidth=2)
plt.show()


# Since y variable is skewed, we will normalise the train df

# In[303]:



print("------Data Standardization------")
sc = StandardScaler()
bike_df_norm = sc.fit_transform(bike_df)
bike_df_norm=pd.DataFrame(bike_df_norm)

# Pairwise Correlation for columns
bike_df_norm.corr()
bike_df_norm.columns = bike_df.columns

print("bike_df_norm->")
print(bike_df_norm)


# # COST FUNCTION AND GRADIENT DESCENT
# 
# 

# In[304]:


def cost_function(x,y,beta):
    # cost_function(X,y,beta):cost of using beta as parameter for LR to fit the data points in X and y
    m = len(x)
    J = np.sum((x.dot(beta)-y)**2)/(2*m)
    return J


#cost1 = cost_function(X,y,beta)
#print(cost1)


# In[305]:


iterations = 2000 #epoch
alpha = 0.005 #learning rate

def gradient_descent(X,y,beta, alpha, iterations):
    # This function updates beta by taking the iterations gradient steps with learning rate
    cost_old = [0] * iterations
    m = len(x)
    for i in range(iterations):
        y_hat = X.dot(beta)
        remaining = y_hat - y
        gradient = X.T.dot(remaining)/m
        beta = beta - (alpha * gradient)
        cost = cost_function(X,y,beta)
        cost_old[i] = cost
    return beta, cost_old


# In[ ]:


sb.pairplot(bike_df_norm)


# # DATA SPLIT AND MODEL TRAINING FOR LR

# In[306]:


# Split the features in X and Y
x = bike_df_norm.drop(columns=['Rented Bike Count'], axis=1)
y = bike_df_norm['Rented Bike Count']


# In[307]:


#Set bias intercept 
x['intercept'] =1


# In[308]:


#Rearrange the index 'intercept'
x = x.reindex(['intercept','Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
       'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Month',
       'Day', 'Weekday', 'Seasons_Autumn', 'Seasons_Spring', 'Seasons_Summer',
       'Seasons_Winter', 'Holiday_Holiday', 'Holiday_No Holiday',
       'Functioning Day_No', 'Functioning Day_Yes'], axis=1)


# In[309]:


#Create test and train data
from sklearn.model_selection import train_test_split
#split the data by percentage
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)


# Since y variable is skewed, we will normalise the train df

# In[310]:


# call gradient_descent function to fit the parameters
# Train Data
beta = np.zeros((X_train.shape[1]))
beta
#beta = np.array([0,0])
(beta_value_new, cost_history) = gradient_descent(X_train, y_train, beta, alpha, iterations)
print(beta_value_new)
print(cost_history)


# In[311]:


#plotting the best fit line
#best_fit_line_x = np.linspace(0,500,500)
#best_fit_line_y = [beta[1]+beta[0]*i for i in best_fit_line_x]

#COST VS EPOCH PLOT
# Cost/Loss Plot per iteration
plt.plot(cost_history)
plt.xlabel("Epoch(Number of Iterations)")
plt.ylabel("Cost(Loss)")
plt.legend(['Alpha : 0.005 '])
plt.show()
print(f'Epoch = {iterations}')
print(f'Learning Rate(Alpha) = {alpha}')
print(f'Lowest cost = {str(np.min(cost_history))}')
print('-------------------------------------------------------------')
print(f'Cost after {iterations} iterations = {str(cost_history[-1])}')


# PREDICT OUTPUT FOR TEST DATA

# In[312]:


# Predict Output for Test Data
y_predicted= np.dot(X_test, beta_value_new)
print("Predicted Y:-")
print(y_predicted)


# EVALUATE TRAINING PERFORMANCE

# In[313]:


# Evaluation Metrics - Calculation of Mean Absolute Error(MAE), Mean Squared Error (MSE), Root Mean Square Error(RMSE), R2 Score for Training
y_pred_train = np.dot(X_train, beta_value_new)
print("------Training Performance Evaluation-------")
print("Mean Absolute Error(MAE)-",mean_absolute_error(y_train,y_pred_train))
print("Mean Squared Error(MSE)-",mean_squared_error(y_train,y_pred_train))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_train,y_pred_train)))
print("R2-",r2_score(y_train,y_pred_train))


# EVALUATE TESTING PERFORMANCE

# In[314]:


# Evaluation Metrics - Calculation of Mean Absolute Error(MAE), Mean Squared Error (MSE), Root Mean Square Error(RMSE), R2 Score for Testing
print("------Testing Performance Evaluation-------")
print("Mean Absolute Error (MAE)-",mean_absolute_error(y_test,y_predicted))
print("Mean Square Error (MSE)-",mean_squared_error(y_test,y_predicted))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_test,y_predicted)))
print("R2-",r2_score(y_test,y_predicted))


# In[307]:


X_train.head()


# # EXPERIMENTATION

# #1. Change Learning Rate

# In[315]:


alpha = 0.5
(beta_value_new_alpha1, cost_history_alpha1)= gradient_descent(X_train, y_train, beta, alpha, iterations)
alpha = 0.05
(beta_value_new_alpha2, cost_history_alpha2) = gradient_descent(X_train, y_train, beta, alpha, iterations)
alpha = 0.001
(beta_value_new_alpha3, cost_history_alpha3)= gradient_descent(X_train, y_train, beta, alpha, iterations)
alpha = 0.025
(beta_value_new_alpha4, cost_history_alpha4)= gradient_descent(X_train, y_train, beta, alpha, iterations)
alpha = 0.00005
(beta_value_new_alpha5, cost_history_alpha5) = gradient_descent(X_train, y_train, beta, alpha, iterations)
alpha = 0.000005
(beta_value_new_alpha6, cost_history_alpha6) = gradient_descent(X_train, y_train, beta, alpha, iterations)

#print(beta_value_new)
#print(cost_history)


# In[316]:


#COST VS EPOCH PLOT
# Cost/Loss Plot per iteration
# Plotting both the curves simultaneously
plt.plot(cost_history_alpha1, color='r', label='0.5')
plt.plot(cost_history_alpha2, color='g', label='0.05')
plt.plot(cost_history_alpha3, color='b', label='0.001')
plt.plot(cost_history, color='b', label='0.005')
plt.plot(cost_history_alpha4, color='#4b0082', label='0.025')
plt.plot(cost_history_alpha5, color='lightcoral', label='0.0005')
plt.plot(cost_history_alpha6, color='k', label='0.00005')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Epoch(Number of Iterations)")
plt.ylabel("Cost(Loss)")
plt.title("Cost Vs Epoch")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()

print(f'Epoch = {iterations}')
print(f'Learning Rate(Alpha) = {alpha}')
print(f'Lowest cost = {str(np.min(cost_history))}')
print('-------------------------------------------------------------')
print(f'Cost after {iterations} iterations = {str(cost_history[-1])}')



# #2. Threshold For Convergence

# In[318]:


# We are checking max iterations at given threshold
# (Jn - Jn+1) / Jn * 100 
alpha = 0.01 #learning rate

def gradient_descent_tolerance(X,y,beta, alpha,tolerance):
    # This function checks the number of iterations till the threshold
    iterations = 0
    cost_old = [] 
    m = len(x)
    while True:
        iterations +=1
        y_hat = X.dot(beta)
        remaining = y_hat - y
        gradient = X.T.dot(remaining)/m
        diff =  - (alpha * gradient)
        beta = beta + diff
        cost_old.append(np.sum((X.dot(beta)-y)**2)/(2*m))
        if len(cost_old) > 1:
            # 2nd - 1st
            cost_percent = ((cost_old[-2] - cost_old[-1]) / cost_old[-2]) * 100
            if cost_percent <= tolerance:
                break
    return beta, iterations


# In[319]:


train_errors = []
test_errors = []
tolerances = [0.0001, 0.001,0.005,0.05,0.02, 0.01, 0.1, 1, 10]
for tolerance in tolerances:
    (beta_value_tolerance_train, iterations_tolerance_train) = gradient_descent_tolerance(X_train, y_train, beta_value_new, alpha,tolerance)
    train_error = cost_function(X_train,y_train,beta_value_tolerance_train ) 
    test_error = cost_function(X_test,y_test,beta_value_tolerance_train )  
    print(f'The Training Error is = {train_error}')
    print(f'The Test Error is = {test_error}')
    print(f'No of Iterations used = {iterations_tolerance_train}')
    print(f'Tolerance =  {tolerance}')
    train_errors.append(train_error)
    test_errors.append(test_error)
#print(beta_value_tolerance_train)
#print(beta_value_tolerance_train)


# In[320]:


#COST VS EPOCH PLOT
# Cost/Loss Plot per iteration
plt.plot(train_errors)
plt.plot(test_errors)
plt.xlabel("Epoch(Number of Iterations)")
plt.ylabel("Cost(Loss)")
plt.legend(['Alpha : 0.005 '])
plt.show()
print(f'Epoch = {iterations}')
print(f'Learning Rate(Alpha) = {alpha}')
print(f'Lowest cost = {str(np.min(cost_history))}')
print('-------------------------------------------------------------')
print(f'Cost after {iterations} iterations = {str(cost_history[-1])}')


# In[321]:



thresholds = [0.0001, 0.001,0.005,0.05,0.02, 0.01, 0.1, 1, 10]
plt.plot(thresholds, train_errors, label = "Train")
plt.plot(thresholds, test_errors, label = "Test")
plt.xscale('log')
plt.xlabel("Threshold of Convergence (%)")
plt.ylabel("Cost")
plt.title("Cost vs. Thresholds of Convergence")
plt.legend()
plt.show()


# #3. Running model with 8 random features

# In[322]:


# Random Variables
# Hour, Visibility, Temperature,Humidity, Dew Point Temperature, Rainfall, Snowfall, Holiday_Holiday, Holiday_NoHoliday
#X_train.info()
X_train_random8 = X_train.drop(columns=['Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)',''
'Month','Day','Weekday','Seasons_Autumn','Seasons_Spring','Seasons_Summer','Seasons_Winter',
'Functioning Day_No','Functioning Day_Yes'],axis=1)

y_train_random8 = y_train.drop(columns=['Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)',
'Month','Day','Weekday','Seasons_Autumn','Seasons_Spring','Seasons_Summer','Seasons_Winter',
'Functioning Day_No','Functioning Day_Yes'],axis=1)

X_test_random8 = X_test.drop(columns=['Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)',''
'Month','Day','Weekday','Seasons_Autumn','Seasons_Spring','Seasons_Summer','Seasons_Winter',
'Functioning Day_No','Functioning Day_Yes'],axis=1)

y_test_random8 = y_test.drop(columns=['Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)',
'Month','Day','Weekday','Seasons_Autumn','Seasons_Spring','Seasons_Summer','Seasons_Winter',
'Functioning Day_No','Functioning Day_Yes'],axis=1)


# In[323]:


alpha = 0.005
iterations = 2000
beta = np.zeros((X_train_random8.shape[1]))
(beta_value_new_random8, cost_history_random8) = gradient_descent(X_train_random8, y_train_random8, beta, alpha, iterations)

#COST VS EPOCH PLOT
# Cost/Loss Plot per iteration
plt.plot(cost_history_random8)
plt.xlabel("Epoch(Number of Iterations)")
plt.ylabel("Cost(Loss)")
plt.legend(['Alpha : 0.005 '])
plt.show()
print(f'Epoch = {iterations}')
print(f'Learning Rate(Alpha) = {alpha}')
print(f'Lowest cost = {str(np.min(cost_history_random8))}')
print('-------------------------------------------------------------')
print(f'Cost after {iterations} iterations = {str(cost_history_random8[-1])}')


# In[324]:


# Predict Output for Test Data
y_predicted= np.dot(X_test_random8, beta_value_new_random8)
print("Predicted Y:-")
print(y_predicted)

# Evaluation Metrics - Calculation of Mean Absolute Error(MAE), Mean Squared Error (MSE), Root Mean Square Error(RMSE), R2 Score for Training
y_pred_train = np.dot(X_train_random8, beta_value_new_random8)
print("------Training Performance Evaluation-------")
print("Mean Absolute Error(MAE)-",mean_absolute_error(y_train_random8,y_pred_train))
print("Mean Squared Error(MSE)-",mean_squared_error(y_train_random8,y_pred_train))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_train_random8,y_pred_train)))
print("R2-",r2_score(y_train_random8,y_pred_train))

# Evaluation Metrics - Calculation of Mean Absolute Error(MAE), Mean Squared Error (MSE), Root Mean Square Error(RMSE), R2 Score for Testing
print("------Testing Performance Evaluation-------")
print("Mean Absolute Error (MAE)-",mean_absolute_error(y_test_random8,y_predicted))
print("Mean Square Error (MSE)-",mean_squared_error(y_test_random8,y_predicted))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_test_random8,y_predicted)))
print("R2-",r2_score(y_test_random8,y_predicted))


# #4. Running model with 8 features that are best suited

# In[325]:


# Suited Variables
#['Intercept','Hour','Dew_point_temperature','Seasons_Winter','Solar_Radiation', 
#'Functioning_Day_Yes','Visibility','Wind_speed','Seasons_Autumn']

# Removing Dew point temperature(°C) as it is having high positive correlation with Temperature(°C) 
# Removing Rainfall(mm) as it is skewed to 0 and based on correlation
# Removing Snowfall (cm) as it is skewed to 0 and based on correlation

# Hour, Wind speed (m/s), Temperature,Humidity, Weekday, Holiday_Holiday, Holiday_NoHoliday,
#'Seasons_Autumn','Seasons_Spring','Seasons_Summer','Seasons_Winter',

# Hour, Wind speed (m/s), Temperature,Humidity, Weekday, Holiday_Holiday, 'Seasons_Summer',
#'Functioning_Day_Yes', Weekday
#X_train.info()
X_train_suited8 =X_train.drop(columns=['Visibility (10m)', 'Dew point temperature(°C)','Solar Radiation (MJ/m2)',
                                        'Rainfall(mm)', 'Snowfall (cm)','Month','Day',
                                        'Functioning Day_No','Seasons_Winter','Seasons_Autumn',
                                         'Seasons_Spring','Holiday_No Holiday'],axis=1)

y_train_suited8 = y_train.drop(columns=['Visibility (10m)', 'Dew point temperature(°C)','Solar Radiation (MJ/m2)',
                                        'Rainfall(mm)', 'Snowfall (cm)','Month','Day',
                                        'Functioning Day_No','Seasons_Winter','Seasons_Autumn',
                                         'Seasons_Spring','Holiday_No Holiday'],axis=1)
X_test_suited8 = X_test.drop(columns=['Visibility (10m)', 'Dew point temperature(°C)','Solar Radiation (MJ/m2)',
                                        'Rainfall(mm)', 'Snowfall (cm)','Month','Day',
                                        'Functioning Day_No','Seasons_Winter','Seasons_Autumn',
                                         'Seasons_Spring','Holiday_No Holiday'],axis=1)

y_test_suited8 =y_test.drop(columns=['Visibility (10m)', 'Dew point temperature(°C)','Solar Radiation (MJ/m2)',
                                        'Rainfall(mm)', 'Snowfall (cm)','Month','Day',
                                        'Functioning Day_No','Seasons_Winter','Seasons_Autumn',
                                         'Seasons_Spring','Holiday_No Holiday'],axis=1)


# In[326]:


alpha = 0.005
iterations = 2000
beta = np.zeros((X_train_suited8.shape[1]))
(beta_value_new_suited8, cost_history_suited8) = gradient_descent(X_train_suited8, y_train_suited8, beta, alpha, iterations)

#COST VS EPOCH PLOT
# Cost/Loss Plot per iteration
plt.plot(cost_history_suited8)
plt.xlabel("Epoch(Number of Iterations)")
plt.ylabel("Cost(Loss)")
plt.legend(['Alpha : 0.005 '])
plt.show()
print(f'Epoch = {iterations}')
print(f'Learning Rate(Alpha) = {alpha}')
print(f'Lowest cost = {str(np.min(cost_history_suited8))}')
print('-------------------------------------------------------------')
print(f'Cost after {iterations} iterations = {str(cost_history_suited8[-1])}')


# In[327]:


# Predict Output for Test Data
y_predicted= np.dot(X_test_suited8, beta_value_new_suited8)
print("Predicted Y:-")
print(y_predicted)

# Evaluation Metrics - Calculation of Mean Absolute Error(MAE), Mean Squared Error (MSE), Root Mean Square Error(RMSE), R2 Score for Training
y_pred_train = np.dot(X_train_suited8, beta_value_new_suited8)
print("------Training Performance Evaluation-------")
print("Mean Absolute Error(MAE)-",mean_absolute_error(y_train_suited8,y_pred_train))
print("Mean Squared Error(MSE)-",mean_squared_error(y_train_suited8,y_pred_train))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_train_suited8,y_pred_train)))
print("R2-",r2_score(y_train_suited8,y_pred_train))

# Evaluation Metrics - Calculation of Mean Absolute Error(MAE), Mean Squared Error (MSE), Root Mean Square Error(RMSE), R2 Score for Testing
print("------Testing Performance Evaluation-------")
print("Mean Absolute Error (MAE)-",mean_absolute_error(y_test_suited8,y_predicted))
print("Mean Square Error (MSE)-",mean_squared_error(y_test_suited8,y_predicted))
print("Root Mean Square Error(RMSE)-",np.sqrt(mean_squared_error(y_test_suited8,y_predicted)))
print("R2-",r2_score(y_test_suited8,y_predicted))


# In[328]:


beta_value_new_suited8


# In[329]:


beta_value_new_random8


# In[330]:


beta_value_new

