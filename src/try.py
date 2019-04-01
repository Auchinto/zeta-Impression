import pandas as pd

data = "../data/Emissions_Full.csv"
df = pd.read_csv(data)

dt = pd.concat([df['ENGINESIZE'],df['CO2EMISSIONS']],axis = 1, keys=['ENGINESIZE','CO2EMISSIONS'])

print(dt)


plt.figure(figsize=(12,4 ))

plt.subplot(2, 5, 1)
plt.hist(df['ENGINESIZE'],color = 'red') ; plt.xlabel("ENGINESIZE") ; plt.ylabel("Frequency")
DF = dict(df['ENGINESIZE'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['ENGINESIZE', 'FREQUENCY'])
dt.sort_values('ENGINESIZE').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\esize.csv',index = False,header = False)



#.to_csv(r'C:\xampp\htdocs\zeta-Impression\data\data.csv')
plt.subplot(2, 5, 2)
plt.hist(df['CYLINDERS'],color = 'green') ; plt.xlabel("CYLINDERS") ; plt.ylabel("Frequency")
DF = dict(df['CYLINDERS'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['CYLINDERS', 'FREQUENCY'])
dt.sort_values('CYLINDERS').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\cyl.csv',index = False,header = False)


plt.subplot(2, 5, 3)
plt.hist(df['City_FC'], color = 'orange'); plt.xlabel("City_FC"); plt.ylabel("Frequency")
DF = dict(df['ENGINESIZE'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['ENGINESIZE', 'FREQUENCY'])
dt.sort_values('ENGINESIZE').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\city_fc.csv',index = False,header = False)


plt.subplot(2, 5, 4)
plt.hist(df['HWY_FC'], color = 'purple'); plt.xlabel("HWY_FC"); plt.ylabel("Frequency")
DF = dict(df['HWY_FC'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['HWY_FC', 'FREQUENCY'])
dt.sort_values('HWY_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\hwy_fc.csv',index = False,header = False)


plt.subplot(2, 5, 5)
plt.hist(df['COMB_FC'], color = 'purple'); plt.xlabel("COMB_FC"); plt.ylabel("Frequency")
DF = dict(df['COMB_FC'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['COMB_FC', 'FREQUENCY'])
dt.sort_values('COMB_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\comb_fc.csv',index = False,header = False)


plt.subplot(2, 5, 6)
plt.hist(df['COMB_MPG_FC'], color = 'purple'); plt.xlabel("COMB_MPG_FC"); plt.ylabel("Frequency")
DF = dict(df['COMB_MPG_FC'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['COMB_MPG_FC', 'FREQUENCY'])
dt.sort_values('COMB_MPG_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\comb_mpg_fc.csv',index = False,header = False)


plt.subplot(2, 5, 7)
plt.hist(df['Diesel'], color = 'brown'); plt.xlabel("Diesel"); plt.ylabel("Frequency")
DF = dict(df['Diesel'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['Diesel', 'FREQUENCY'])
dt.sort_values('Diesel').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\diesel.csv',index = False,header = False)


plt.subplot(2, 5, 8)
plt.hist(df['Ethanol'], color = 'brown'); plt.xlabel("Ethanol"); plt.ylabel("Frequency")
DF = dict(df['Ethanol'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['Ethanol', 'FREQUENCY'])
dt.sort_values('Ethanol').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\ethanol.csv',index = False,header = False)


plt.subplot(2, 5, 9)
plt.hist(df['Gasoline'], color = 'brown'); plt.xlabel("Gasoline"); plt.ylabel("Frequency")
DF = dict(df['Gasoline'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['Gasoline', 'FREQUENCY'])
dt.sort_values('Gasoline').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\gasoline.csv',index = False,header = False)


plt.subplot(2, 5, 10)
plt.hist(df['Prem_Gasoline'], color = 'brown'); plt.xlabel("Premium_Gasoline"); plt.ylabel("Frequency")
DF = dict(df['Prem_Gasoline'].value_counts(sort=True,ascending = True))
dt = pd.DataFrame(list(DF.items()), columns=['Prem_Gasoline', 'FREQUENCY'])
dt.sort_values('Prem_Gasoline').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\prem_gas.csv',index = False,header = False)


plt.tight_layout()
plt.show()





dt = pd.concat([df['ENGINESIZE'],df['CO2EMISSIONS']],axis = 1, keys=['ENGINESIZE','CO2EMISSIONS'])
dt.sort_values('ENGINESIZE').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\esize_c.csv',index = False,header = False)

dt = pd.concat([df['CYLINDERS'],df['CO2EMISSIONS']],axis = 1, keys=['CYLINDERS','CO2EMISSIONS'])
dt.sort_values('CYLINDERS').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\cyl_c.csv',index = False,header = False)

dt = pd.concat([df['City_FC'],df['CO2EMISSIONS']],axis = 1, keys=['City_FC','CO2EMISSIONS'])
dt.sort_values('City_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\city_fc_c.csv',index = False,header = False)

dt = pd.concat([df['HWY_FC'],df['CO2EMISSIONS']],axis = 1, keys=['HWY_FC','CO2EMISSIONS'])
dt.sort_values('HWY_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\hwy_fc_c.csv',index = False,header = False)

dt = pd.concat([df['COMB_FC'],df['CO2EMISSIONS']],axis = 1, keys=['COMB_FC','CO2EMISSIONS'])
dt.sort_values('COMB_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\comb_fc_c.csv',index = False,header = False)

dt = pd.concat([df['COMB_MPG_FC'],df['CO2EMISSIONS']],axis = 1, keys=['COMB_MPG_FC','CO2EMISSIONS'])
dt.sort_values('COMB_MPG_FC').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\comb_mpg_fc_c.csv',index = False,header = False)

dt = pd.concat([df['Diesel'],df['CO2EMISSIONS']],axis = 1, keys=['Diesel','CO2EMISSIONS'])
dt.sort_values('Diesel').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\diesel_c.csv',index = False,header = False)

dt = pd.concat([df['Ethanol'],df['CO2EMISSIONS']],axis = 1, keys=['Ethanol','CO2EMISSIONS'])
dt.sort_values('Ethanol').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\ethanol_c.csv',index = False,header = False)

dt = pd.concat([df['Gasoline'],df['CO2EMISSIONS']],axis = 1, keys=['Gasoline','CO2EMISSIONS'])
dt.sort_values('Gasoline').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\gasoline_c.csv',index = False,header = False)

dt = pd.concat([df['Prem_Gasoline'],df['CO2EMISSIONS']],axis = 1, keys=['Prem_Gasoline','CO2EMISSIONS'])
dt.sort_values('Prem_Gasoline').to_csv(r'C:\xampp\htdocs\zeta-Impression\data\prem_gas_c.csv',index = False,header = False)
