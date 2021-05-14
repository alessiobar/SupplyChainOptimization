import pandas as pd
import numpy as np
import statistics
import os

#Dataset
os.chdir(r"C:\Users\alema\Downloads")
df = pd.read_csv("Supply_Chain_Shipment_Pricing_Data (1).csv")
#print(df[x].value_counts()) 

#Replacing NA with the mode for categorical columns
for x in df.columns: 
    try: #this aims at determining the column type by attempting to cast the first element(s) 
        for y in range(10): 
            df[x][y] = int(df[x][y])
    except Exception: #failed columns are most likely categorical, to be sure just set len(df[column]) instead of 10 in the range
        df.fillna(df[x].mode())

#Rename some columns for easiness
df.rename(columns = {'shipment mode':'mode', "scheduled delivery date":"date","line item quantity":"quantity", 'freight cost (usd)':"cost",'weight (kilograms)':"weight"}, inplace = True)

#Convert numerals in the weight column to integer format
for x in range(len(df["id"])):
    try:
        df["weight"][x] = int(df["weight"][x])
    except Exception:
        pass

#The only columns having engineered missing values where cost and weight, here non-fixable rows are dropped
to_drop = []
for e in ["weight","cost"]:
    for j in range(len(df[e])):
        try:
            if str(df[e][j])[0] == "S":
                the_id = df[e][j][df[e][j].find(":") + 1:df[e][j].find(")")]
                row = df.iloc[[x for x in range(len(df["id"])) if df["id"][x]==int(the_id)][0]]
                print("the row: "+ str(row[e]))
                i=0
                while str(row)[0]== "S":
                    i+=1
                    if i==100: 
                        print("infinite loop?")
                        break
                    the_id = df[e][j][df[e][j].find(":") + 1:df[e][j].find(")")]
                    row = df.iloc[[x for x in range(len(df["id"])) if df["id"][x] == int(the_id)][0]]
                if str(row)[0]== "W" or str(row)[0]== "I" or str(row)[0]== "F" :
                    pass
                else: 
                    df[e][j] = row[e]

        except Exception:
            break
        try:
            df[e][j] = int(df[e][j])

        except Exception:
            if e=="cost":
                if j not in to_drop:
                    to_drop.append(j)
            if e=="weight":
                df[e][j] = "placeholder"

#Substituting ex-post the missing values with the overall weight_mean, computed ex-ante on the non missing values
weight_mean = round(statistics.mean([x for x in df["weight"] if isinstance(x, int)]), 2)
for x in range(len(df["cost"])):
    if df["weight"][x]=="placeholder":
        df["weight"][x] = weight_mean

#Dropping elements added in the to_drop list
to_drop = list(set(to_drop))
for x in to_drop:
    df.drop(x, inplace=True)
df = df.reset_index(drop=True)

#Dropping id because it is useless for the analysis part
df =df.drop(columns=["id"])

#Replacing outliers with Q1 and Q3
classes = ["cost",'weight', "cost",'pack price', 'unit price','unit of measure (per pack)','line item value']
for f in classes:
    q1, q3 = df[f].describe().values[4], df[f].describe().values[-2]
    iQR = q3 - q1
    for p in range(len(df["weight"])):
        if df[f][p] <= q1-1.5*iQR or df[f][p]>= q3+1.5*iQR: #IQR method
            if df[f][p] <= q1-1.5*iQR:
                df[f][p] = q1
            else:
                df[f][p] = q3

#Dummy Variables          
df = pd.get_dummies(df, columns=['project code', 'pq #', 'po / so #', 'asn/dn #', 'country',
       'managed by', 'fulfill via', 'vendor inco term', 'mode',
       'pq first sent to client date', 'po sent to vendor date', 'date',
       'delivered to client date', 'delivery recorded date', 'product group',
       'sub classification', 'vendor', 'item description',
       'molecule/test type', 'brand', 'dosage form',
        'manufacturing site',
       'first line designation',"date"])

df["dosage"] = [df["dosage"].mode()[0] if isinstance(df["dosage"][x], float) else df["dosage"][x] for x in range(len(df["weight"]))]
df = pd.get_dummies(df, columns=['dosage'])

#Split into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["cost"]), df["cost"], test_size=0.33, random_state=42)

#Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Grid for GridSearchCV
grid_rf = {'n_estimators': [200,1000],
               'max_depth': [10, 100],
               'min_samples_split':[2, 10],
               'min_samples_leaf': [1, 4],
               'bootstrap': [True, False]}

#GridSearch and Model Fit
something_rf = GridSearchCV(rf, param_grid=grid_rf, cv=5, verbose=2)
something_rf.fit(df.drop(X_train, y_train) 
something_rf.best_params_
predictions = something.predict(X_test)
errors = abs(predictions - y_test)
                 
mean_squared_error(y_test, predictions, squared=True)
mean_absolute_error(y_test, predictions)
scores_rf = cross_val_score(RandomForestRegressor(random_state = 42, n_estimators= 1000, min_samples_split =2, min_samples_leaf = 1, max_depth= 100, bootstrap= True), df.drop(columns=["cost"]), df["cost"], cv=10 ) #df.drop(columns=["cost"]), df["cost"], cv=10)
scores_rf.mean()       
                 
#rf = RandomForestRegressor(random_state = 42, n_estimators= 200, min_samples_split =2, min_samples_leaf = 1, max_depth= 100, bootstrap= True)
                 
#Extreme Gradient Boosting
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score                 
aaa=list(df.columns.values)
i=0
for a in aaa: #remove strange characters for allowing xgb to work
    for char in ["[","]","<"]:
        aa = a.split(char)
        if len(aa)>1:
            a = ""
            for y in aa:
               if y!="":
                   a+=y
            aa = a
        else:
            aa = aa[0]
    aaa[i] = aa
    i += 1
                 
i=0
for x in aaa:
    df.rename(columns = {list(df.columns.values)[i]: x}, inplace=True)
    i+=1

#Remove duplicate columns that appearently appeared(?)
duplicate_columns = df.columns[df.columns.duplicated()]
df = df.loc[:,~df.columns.duplicated()]

#Grid for XGBoost
grid_xgb={'max_depth': [1, 3],
          'gamma': [1,10],
          'min_child_weight' : [1, 10],
          'n_estimators': [200,1000],
       }

something_xgb = GridSearchCV(XGBRegressor(random_state=42), param_grid=grid_xgb, cv=5, verbose=2)
something_xgb.fit(X_train, y_train) #df.drop(columns=["cost"]), df["cost"]
something_xgb.best_params_

scores_xgb = cross_val_score(XGBRegressor(random_state=42, max_depth=3, gamma=10, min_child_weight=1,n_estimators=1000), df.drop(columns=["cost"]), df["cost"], cv=10)
scores_xgb.mean()
predictions = something_xgb.predict(X_test)
errors = abs(predictions - y_test)
mean_squared_error(y_test, predictions)

#Find top n features selected by the model
xaz = list(something_xgb.feature_importances_)
res=[]
for x in range(10):
    m = max(xaz)
    print(m)
    i=0
    for y in xaz:
        if y == m:
            res.append((list(df.columns.values)[i],y))
            xaz.pop(i)
        i+=1
pog.pop()
pog = list(df.columns.values)
asd = pd.DataFrame(xaz)
asd.transpose()

#Refit di xgboost on the whole initial dataset
df111 = df
df111.to_numpy()
something_xgb.fit(df111.drop(columns=["cost"]), df111["cost"])

#Optimization Function applied to shipment mode and constructed for XGBoost
for y in range(100):
    for x in ["mode_Air", "mode_Air Charter", "mode_Ocean", "mode_Truck"]:
        new_ob = df111.iloc[y]
        new_ob["mode_Air"], new_ob["mode_Air Charter"], new_ob["mode_Ocean"], new_ob["mode_Truck"] = 0,0,0,0
        new_ob[x] = 1
        original = new_ob["cost"]
        new_ob = new_ob.drop("cost")
        new_ob = new_ob.to_frame()
        new_ob.reset_index(inplace=True)
        new_ob = new_ob.transpose()
        new_ob=new_ob.rename(columns=new_ob.iloc[0])
        new_ob=new_ob.drop(["index"])
        new_ob.drop(columns=["na"], inplace=True)
        for col in new_ob.columns:
            new_ob[col] = pd.to_numeric(new_ob[col], errors='coerce')
        predictions=something_xgb.predict(new_ob)
        print(predictions)

# Some Plots
import seaborn as sn
import matplotlib.pyplot as plt

data=df[["mode_Ocean","country_Namibia","cost","weight"]].corr()
dfe = pd.DataFrame(data,columns=['mode_Ocean','country_Namibia','cost',"weight"])
corrMatrix = dfe.corr()
sn.heatmap(corrMatrix, annot=True, )
plt.show()

#Group 12               
