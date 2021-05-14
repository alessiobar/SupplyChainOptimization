import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import os
import matplotlib as plt
os.chdir(r"C:\Users\alema\Downloads")
import statistics


df1 = pd.read_csv("Supply_Chain_Shipment_Pricing_Data (1).csv")
df = df1
#print(df[x].value_counts())

df['line item insurance (usd)']=0

for x in df.columns: #should be only for categorical, so subset it to be good enough generalizable code..
    try:
        df[x][0] = int(df[x][0])
    except Exception:
        df.fillna(df[x].mode())

df.rename(columns = {'shipment mode':'mode', "scheduled delivery date":"date","line item quantity":"quantity", 'freight cost (usd)':"cost",'weight (kilograms)':"weight"}, inplace = True)
df1 = df
to_drop = []

for x in range(len(df["id"])):
    try:
        df["weight"][x] = int(df["weight"][x])
    except Exception:
        pass

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
                else: # aka if it's a number, lo pluggo
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


weight_mean = round(statistics.mean([x for x in df["weight"] if isinstance(x, int)]), 2)
for x in range(len(df["cost"])):
    if df["weight"][x]=="placeholder":
        df["weight"][x] = weight_mean

#to_drop = np.load("to_drop.npy")
to_drop = list(set(to_drop))
for x in to_drop:
    df.drop(x, inplace=True)
df = df.reset_index(drop=True)

df =df.drop(columns=["id"])

#df.to_excel("basta.xlsx")
#df1 = pd.read_excel("intermezzo.xlsx")

classes = ["cost",'weight', "cost",'pack price', 'unit price','unit of measure (per pack)','line item value']
# for y in classes:
#     for x in range(len(df["cost"])):
#         try:
#             df[y][x] = int(df["weight"][x])
#         except Exception:
#             pass

for f in classes:
    q1, q3 = df[f].describe().values[4], df[f].describe().values[-2]
    iQR = q3 - q1
    for p in range(len(df["weight"])):
        if df[f][p] <= q1-1.5*iQR or df[f][p]>= q3+1.5*iQR:
            if df[f][p] <= q1-1.5*iQR:
                df[f][p] = q1
            else:
                df[f][p] = q3

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

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["cost"]), df["cost"], test_size=0.33, random_state=42)

#RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score


grid_rf = {'n_estimators': [200,1000],
               'max_depth': [10, 100],
               'min_samples_split':[2, 10],
               'min_samples_leaf': [1, 4],
               'bootstrap': [True]}#, False]}

rf = RandomForestRegressor(random_state = 42)#, n_estimators= 200, min_samples_split = 2, min_samples_leaf = 2, max_features ='auto', max_depth= 80, bootstrap= True)
#rf.fit(X_train, y_train)

#rf_random = rf
something_rf = GridSearchCV(rf, param_grid=grid_rf, cv=5, verbose=2) # e no, io devo farla su tutto poi, per tunare a modo, non solo sullo split.
something_rf.fit(df.drop(columns=["cost"]), df["cost"]) # e no devo tunare sulo su train split
something_rf.best_params_

#rf = RandomForestRegressor(random_state = 42, n_estimators= 200, min_samples_split =2, min_samples_leaf = 1, max_depth= 100, bootstrap= True)

predictions = something.predict(X_test)
errors = abs(predictions - y_test)
mean_squared_error(y_test, predictions)
mean_squared_error(y_test, predictions, squared=False)
                                        #187273236.05130753 co 463 obs
                                        #186722576.5434092
                                        # 46421417.5825533 co 690 obs
                                        # 46343852.48890569 wout normalization/scaling
                                        #  6399807.202095908 co 690 obs retrainando
                                        #  6399797.518667495 wout normalization
                                        #39475272.091516696
                                        # 5636978.805694842
#                                        17794565.04079134
                                        #17624560
mean_absolute_error(y_test, predictions)

#retrain
rf.fit(df.drop(columns=["cost"]), df["cost"])

scores_rf = cross_val_score(RandomForestRegressor(random_state = 42, n_estimators= 1000, min_samples_split =2, min_samples_leaf = 1, max_depth= 100, bootstrap= True), df.drop(columns=["cost"]), df["cost"], cv=10 ) #df.drop(columns=["cost"]), df["cost"], cv=10)
scores_rf.mean()

####XGBOOST
aaa=list(df.columns.values)
i=0
for a in aaa:
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

duplicate_columns = df.columns[df.columns.duplicated()]
df = df.loc[:,~df.columns.duplicated()]

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
grid_xgb={'max_depth': [1, 3],
          'gamma': [1,10],
          'min_child_weight' : [1, 10],
          'n_estimators': [200,1000],
       }

something_xgb = GridSearchCV(XGBRegressor(random_state=42), param_grid=grid_xgb, cv=5, verbose=2) # e no, io devo farla su tutto poi, per tunare a modo, non solo sullo split.
something_xgb.fit(df.drop(columns=["cost"]), df["cost"]) #di nuovo no
something_xgb.best_params_

scores_xgb = cross_val_score(XGBRegressor(random_state=42, max_depth=3, gamma=10, min_child_weight=1,n_estimators=1000), df.drop(columns=["cost"]), df["cost"], cv=10)
scores_xgb.mean()

something_xgb=XGBRegressor(random_state=42,max_depth=3, gamma=10, min_child_weight=1,n_estimators=1000)
something_xgb.fit(X_train, y_train)
predictions = something_xgb.predict(X_test)
errors = abs(predictions - y_test)
mean_squared_error(y_test, predictions)




 # with open('your_file.txt', 'w') as f:
 #     for item in olihhh: #list(df.columns.values):
 #         f.write("%s\n" % item)





xaz = list(something_xgb.feature_importances_)

res=[]
for x in range(10):
    m = max(xaz)
    print(m)
    i=0
    for y in xaz:
        if y == m:
            print("gy")
            res.append((list(df.columns.values)[i],y))
            xaz.pop(i)
        i+=1



pog.pop()
pog = list(df.columns.values)

asd = pd.DataFrame(xaz)
asd.transpose()





#refit di xgboost
df111 = df

df111.to_numpy()
something_xgb.fit(df111.drop(columns=["cost"]), df111["cost"])

for y in range(100):
    print("--------------")
    for x in ["mode_Air", "mode_Air Charter", "mode_Ocean", "mode_Truck"]:
        #print(x)
        new_ob = df111.iloc[y]
        #print(new_ob[x])
        new_ob["mode_Air"], new_ob["mode_Air Charter"], new_ob["mode_Ocean"], new_ob["mode_Truck"] = 0,0,0,0
        new_ob[x] = 1
        #print(new_ob[x])
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





for x in range(100):

    new_ob["mode_Truck"] = 0
    new_ob["mode_Air"] = 1
    yy = new_ob["cost"]
    new_ob = new_ob.drop("cost")
    predictions = something_xgb.predict([new_ob])
    errors = abs(predictions - yy)
    print(errors)

predictions
new_ob["mode_Truck"] = 1
new_ob["mode_Air"] = 0

df.to_excel("witholeh.xlsx")

def optimizer(obs):
    truevalue = obs["cost"]
    obs = obs.drop("cost")
    ante_pred = rf_random.predict([obs])
    print(ante_pred)
    df["mode_Air"] = int(input())
    df["mode_Truck"] = int(input())
    post_pred = rf_random.predict([obs])
    if ante_pred > post_pred:
        return obs
    else:
        return new_obs

print(optimizer(df.iloc[0]))

#################################
import matplotlib.pyplot as plt
X_grid = [1,2,3,4,5]

# Scatter plot for original data
plt.scatter(y_test, predictions, color='blue')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid),
         color='green')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# rectangular box plot
bplot1 = ax1.boxplot(df1,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=["1","2","3","4"])  # will be used to label x-ticks

ax1.set_title('Rectangular box plot')
plt.show()




from plotnine import * #attento al df1
(
    ggplot(df1)
    + geom_boxplot(aes(x="mode", y='cost'))
    + scale_x_discrete(labels=["Air","Truck","Air Charter", "Ocean"], name='boh')  # change ticks labels on OX
)


import matplotlib.pyplot as plt
plt.style.use('ggplot')

x = ['Air', 'Air Charter', 'Ocean', 'Truck']
energy = [393, 677, 465, 409]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color=["blue","darkblue","red","darkred"])
plt.xlabel("Shipment Mode")
plt.ylabel("Predicted Freight Cost")
plt.title("Freight Cost Optimization")

plt.xticks(x_pos, x)


plt.show()



data=df[["mode_Ocean","country_Namibia","cost","weight"]].corr()
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt



dfe = pd.DataFrame(data,columns=['mode_Ocean','country_Namibia','cost',"weight"])

corrMatrix = dfe.corr()
sn.heatmap(corrMatrix, annot=True, )
plt.show()