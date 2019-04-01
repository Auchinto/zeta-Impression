import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pylab as pl
import numpy as np
from scipy import stats

data = "../data/Emissions_Full.csv"
df = pd.read_csv(data)

df.describe()

missing_data = df.isnull()

df.rename(columns={'FUELCONSUMPTION_CITY':'City_FC', 'FUELCONSUMPTION_HWY':'HWY_FC','FUELCONSUMPTION_COMB':'COMB_FC','FUELCONSUMPTION_COMB_MPG':'COMB_MPG_FC'}, inplace=True)

dummy_fuel = pd.get_dummies(df['FUELTYPE'])
#dummy_fuel.head()
#dummy_fuel.describe()

dummy_fuel.rename(columns={
    'D':'Diesel', 'E':'Ethanol','X':'Gasoline','Z':'Prem_Gasoline'
}, inplace=True)
dummy_fuel.head()
#merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_fuel], axis=1)

# drop original column "fuel-type" from "df"
df.drop("FUELTYPE", axis = 1, inplace=True)

#TO BE ADDED HERE #1
plt.figure(figsize=(12,4 ))

plt.subplot(2, 5, 1)
plt.hist(df['ENGINESIZE'],color = 'red') ; plt.xlabel("ENGINESIZE") ; plt.ylabel("Frequency")




#.to_csv(r'C:\xampp\htdocs\zeta-Impression\data\data.csv')
plt.subplot(2, 5, 2)
plt.hist(df['CYLINDERS'],color = 'green') ; plt.xlabel("CYLINDERS") ; plt.ylabel("Frequency")



plt.subplot(2, 5, 3)
plt.hist(df['City_FC'], color = 'orange'); plt.xlabel("City_FC"); plt.ylabel("Frequency")



plt.subplot(2, 5, 4)
plt.hist(df['HWY_FC'], color = 'purple'); plt.xlabel("HWY_FC"); plt.ylabel("Frequency")


plt.subplot(2, 5, 5)
plt.hist(df['COMB_FC'], color = 'purple'); plt.xlabel("COMB_FC"); plt.ylabel("Frequency")



plt.subplot(2, 5, 6)
plt.hist(df['COMB_MPG_FC'], color = 'purple'); plt.xlabel("COMB_MPG_FC"); plt.ylabel("Frequency")



plt.subplot(2, 5, 7)
plt.hist(df['Diesel'], color = 'brown'); plt.xlabel("Diesel"); plt.ylabel("Frequency")



plt.subplot(2, 5, 8)
plt.hist(df['Ethanol'], color = 'brown'); plt.xlabel("Ethanol"); plt.ylabel("Frequency")



plt.subplot(2, 5, 9)
plt.hist(df['Gasoline'], color = 'brown'); plt.xlabel("Gasoline"); plt.ylabel("Frequency")



plt.subplot(2, 5, 10)
plt.hist(df['Prem_Gasoline'], color = 'brown'); plt.xlabel("Premium_Gasoline"); plt.ylabel("Frequency")


plt.tight_layout()
plt.show()





adf = df[['ENGINESIZE','CYLINDERS','City_FC','HWY_FC','COMB_FC','COMB_MPG_FC',
          'Gasoline','Prem_Gasoline','CO2EMISSIONS']]

corr = adf.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True) 
plt.show()


def calculate_pvalues(adf):
    df = adf.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=adf.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
    return pvalues

df_corr = adf.corr()
pval = calculate_pvalues(adf) 
# create three masks for p-values
r1 = df_corr.applymap(lambda x: '{:.2f}*'.format(x)) 
r2 = df_corr.applymap(lambda x: '{:.2f}**'.format(x))
r3 = df_corr.applymap(lambda x: '{:.2f}***'.format(x))
r4 = df_corr.applymap(lambda x: '{:.2f}'.format(x))
# applying these masks to the correlation matrix
df_corr = df_corr.mask(pval>.1,r4)
df_corr = df_corr.mask(pval<=.1,r1)
df_corr = df_corr.mask(pval<=.05,r2)
df_corr = df_corr.mask(pval<=.01,r3)

import itertools
correlations = {}
columns = adf.columns.tolist()

for col_a, CO2EMISSIONS in itertools.combinations(columns, 2):
    correlations[col_a] = stats.pearsonr(adf.loc[:, col_a], adf.loc[:, CO2EMISSIONS])

result = adf.from_dict(correlations, orient='index')
result.columns = ['PCC', 'p-value']
rho = result.sort_index()
print(result.sort_index())

pearson_coef, p_value = stats.pearsonr(df["Prem_Gasoline"], df["CO2EMISSIONS"])
print("Coeff:", pearson_coef, " \nP-value of", p_value)

sns.pairplot(adf, x_vars=["COMB_MPG_FC","Gasoline","City_FC"], 
             y_vars="CO2EMISSIONS", kind = 'reg')
sns.pairplot(adf, x_vars=["ENGINESIZE","CYLINDERS",
                          "HWY_FC","Prem_Gasoline"], y_vars="CO2EMISSIONS", kind = 'reg')


from sklearn.model_selection import train_test_split as tts

# Split data into test (30%) and training (70%) sets
train_data, test_data = tts(adf,train_size=0.7)
#print(type(train_data))
print("Training samples", train_data.shape[0])
print("Testing samples: ", test_data.shape[0])


sns.pairplot(train_data, x_vars=["COMB_MPG_FC","Gasoline","City_FC"], 
             y_vars="CO2EMISSIONS", kind = 'reg')
sns.pairplot(train_data, x_vars=["ENGINESIZE","CYLINDERS",
                          "HWY_FC","Prem_Gasoline"], y_vars="CO2EMISSIONS", kind = 'reg')

from sklearn.linear_model import LinearRegression as lreg
line = lreg()
x = train_data[['COMB_MPG_FC']]
y = train_data.CO2EMISSIONS

#print(x)
#print(y)

line.fit(x,y) # Train the model with the training data set

# The coefficients
#zip(features, line.intercept_)


plt.scatter(x, y,  color='blue')
reg_line = line.coef_*x + line.intercept_
#dt = pd.concat([x,y],axis = 1, keys=['COMB_MPG_FC','CO2EMISSIONS'])
#dt.to_csv(r'C:\xampp\htdocs\zeta-Impression\data\lr_1.csv',index = False,header = False)
#print(dt)
#dt = pd.concat([x,reg_line['COMB_MPG_FC']],axis = 1, keys=['COMB_MPG_FC','CO2EMISSIONS'])
#dt.sort_values('COMB_MPG_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\lr_2.csv',index = False,header = False)
#print(dt)
plt.plot(x, reg_line, '-r')
plt.xlabel("COMB_MPG_FC")
plt.ylabel("Emissions")


from sklearn import metrics

x_test = test_data[['COMB_MPG_FC']]
y_test = test_data.CO2EMISSIONS

y_pred = pd.Series(line.predict(x_test)) # Calculate R-sqrd for the model using test data set

def DistributionPlot(x_test,RedFunction,BlueFunction,RedName,BlueName,Title ): # 
    width = 4 ; height = 4
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="red", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="blue", label=BlueName, ax=ax1)
    #print(type(x_test),type(RedFunction),type(BlueFunction))
    #dt = pd.concat([x_test,RedFunction],axis = 1, keys=['COMB_MPG_FC','CO2EMISSIONS'])
    
    #dt.to_csv(r'C:\xampp\htdocs\zeta-Impression\data\dis_1.csv',index = False,header = False)

    #dt = pd.concat([x_test,BlueFunction],axis = 1, keys=['COMB_MPG_FC','CO2EMISSIONS'])
    #dt.to_csv(r'C:\xampp\htdocs\zeta-Impression\data\dis_2.csv',index = False,header = False)
    
    plt.title(Title)
    plt.xlabel("COMB_MPG_FC")
    plt.ylabel("CO2 Emissions")
    plt.show() ; plt.close()

Title="Distribution Plot: Predictions using training data vs training data distribution"
DistributionPlot(x_test,y_test,y_pred,"Actual Values","Predicted Values",Title)


print("R-squared: %.2f" % line.score(x,y)) # Calculate R-sqrd for the model using test data set
print("MAE: %.2f" % metrics.mean_absolute_error(y_test, y_pred))
print("MSE:%.2f" % metrics.mean_squared_error(y_test, y_pred))
print("RMSE:%.2f" %  np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

features = ['ENGINESIZE','CYLINDERS','City_FC','HWY_FC','COMB_FC','COMB_MPG_FC']
slr_evaluation = {'Feature': features,
#'Mean absolute error': [22.38,25.90,20.26,23.60,20.70,18.06],
'MSE x10': [85.989,113.18,81.954,100.843,83.780,72.808],
'R2-score (%)': [76,73,81,74,80,82]}
slr_df = pd.DataFrame(slr_evaluation
                      , columns=['Feature', 
                                 #'Mean absolute error',
                                 'MSE x10','R2-score (%)'])
df = slr_df.groupby(['Feature'])[
    #'Mean absolute error',
    'MSE x10','R2-score (%)'].sum() 
df.plot(kind='barh') #(kind='bar')
