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


df1 = pd.read_csv("Supply_Chain_Shipment_Pricing_Data (1).csv")
df = df1
#for x in df.columns:
#    print(df[x].value_counts())

df['line item insurance (usd)']=0
for x in df.columns:
    try:
        df[x][0] = int(df[x][0])
    except Exception:
        df.fillna(df[x].mode())

#df = df1[["id","country","shipment mode",'scheduled delivery date', 'weight (kilograms)', 'line item quantity', 'freight cost (usd)']]

# seasons = {"Winter":["Jan","Feb","Dec"], "Spring":["Mar","Apr","May"], "Summer":["Jun","Jul","Aug"], "Autumn":["Sep","Oct","Nov"]}
# l = ["Winter","Spring","Summer","Autumn"]
#
# for x in range(len(df["country"])):
#     for y in l:
#         a = df["scheduled delivery date"][x]
#         a = a[a.find("-")+1:]
#         a = a[:a.find("-")]
#         if a in seasons[y]:
#             df["scheduled delivery date"][x] = y

df.rename(columns = {'shipment mode':'mode', "scheduled delivery date":"date","line item quantity":"quantity", 'freight cost (usd)':"cost",'weight (kilograms)':"weight"}, inplace = True)
df1 = df
to_drop = []

##############################              già fatto --> np.save("to_drop", np.array(to_drop)) #salvato l'array
for e in ["weight","cost"]:
    for j in range(len(df[e])):
        #print(j)
        #print(df.iloc[j])
        #print(str(df[e][j]))
        try:
            if str(df[e][j])[0] == "S":
                #print("non puo esserci secondo except dopo")
                the_id = df[e][j][df[e][j].find(":") + 1:df[e][j].find(")")]
                #print("the id:" + str(the_id))
                row = df.iloc[[x for x in range(len(df["id"])) if df["id"][x]==int(the_id)][0]]
                print("the row: "+ str(row[e]))
                i=0
                while str(row)[0]== "S": #devo ripetere il processo :DDD
                    i+=1
              #      print("Trial #2toInfinity")
                    if i==100:
                        print("infinite loop?")
                        break
                    the_id = df[e][j][df[e][j].find(":") + 1:df[e][j].find(")")]
             #       print("the id:" + str(the_id))
                    row = df.iloc[[x for x in range(len(df["id"])) if df["id"][x] == int(the_id)][0]]
            #        print("the row: " + str(row[e]))
                if str(row)[0]== "W": #droppo la row, mi so rotto i coglioni, non devo fare nulla
                    pass
                elif str(row)[0]== "I":  #sta classe vale solo per il freight
                    pass #droppo la row, quindi nulla
                elif str(row)[0]== "F": #sta classe vale solo per il freight
                    pass #droppo la row, quindi nulla
                else: # aka if it's a number, lo pluggo
                    df[e][j] = row[e]

        except Exception:
           # print("qualcosa è andato a puttane")
            break
        try:
           # print("cazzula")
            df[e][j] = int(df[e][j])

        except Exception:
            if j not in to_drop:
                to_drop.append(j)

            #df.drop(j, inplace=True)
    #df = df.reset_index(drop=True)

to_drop = np.load("to_drop.npy")
to_drop = list(set(to_drop))
for x in to_drop:
    df.drop(x, inplace=True) #690 porcodaus manco 1000
df = df.reset_index(drop=True)

df =df.drop(columns=["id"])

#df = pd.get_dummies(df, columns=["country","mode","date"])
#df.to_excel("intermezzo.xlsx")







df1 = pd.read_excel("intermezzo.xlsx")

classes = ["cost",'weight', "cost",'pack price', 'unit price','unit of measure (per pack)','line item value']
#for j in ["mode_Air","mode_Truck","mode_Air Charter","mode_Ocean"]:
for f in classes:
    #good_ids = [x for x in range(len(df["weight"])) if int(df[j][x]) == 1]
    print(f)
    #olehh = pd.DataFrame([df[f][x] for x in good_ids]).describe()
    #q1, q3 = olehh.values[4], olehh.values[-2]
    q1, q3 = df[f].describe().values[4], df[f].describe().values[-2]
    iQR = q3 - q1
    for p in range(len(df["weight"])):
        if df[f][p] <= q1-1.5*iQR or df[f][p]>= q3+1.5*iQR:
            if df[f][p] <= q1-1.5*iQR:
                df[f][p] = q1
            else:
                df[f][p] = q3


#df = pd.get_dummies(df, columns=['mode']) #perche voglio farlo per colonna per qualche motivo

df = pd.get_dummies(df, columns=['project code', 'pq #', 'po / so #', 'asn/dn #', 'country',
       'managed by', 'fulfill via', 'vendor inco term', 'mode',
       'pq first sent to client date', 'po sent to vendor date', 'date',
       'delivered to client date', 'delivery recorded date', 'product group',
       'sub classification', 'vendor', 'item description',
       'molecule/test type', 'brand', 'dosage form',
        'manufacturing site',
       'first line designation',"date"])


#print(df["dosage"].value_counts())
#gio = ["alessio" if isinstance(df["dosage"][x], float) else df["dosage"][x] for x in range(len(df["weight"]))] #NA Percentage
#gioo=len([x for x in gio if x=="alessio"])
df["dosage"] = [df["dosage"].mode()[0] if isinstance(df["dosage"][x], float) else df["dosage"][x] for x in range(len(df["weight"]))]
df = pd.get_dummies(df, columns=['dosage']) #???

#scaler = StandardScaler().fit(df[["weight","quantity"]])
#X_scaled = pd.DataFrame(scaler.transform(df[["weight","quantity"]]))
#df["weight"], df["quantity"] = X_scaled[0], X_scaled[1]

# X_train, X_test, y_train, y_test = train_test_split(df[['quantity', 'weight', 'country_Botswana', 'country_Burundi',
#        'country_Cameroon', 'country_Congo, DRC', "country_Côte d'Ivoire",
#        'country_Dominican Republic', 'country_Ethiopia', 'country_Ghana',
#        'country_Guyana', 'country_Haiti', 'country_Kenya',
#        'country_Mozambique', 'country_Namibia', 'country_Nigeria',
#        'country_Pakistan', 'country_Rwanda', 'country_Senegal',
#        'country_Sierra Leone', 'country_South Africa', 'country_South Sudan',
#        'country_Tanzania', 'country_Uganda', 'country_Vietnam',
#        'country_Zambia', 'country_Zimbabwe', 'mode_Air', 'mode_Air Charter',
#        'mode_Ocean', 'mode_Truck', 'date_Autumn', 'date_Spring', 'date_Summer',
#        'date_Winter']], df["cost"], test_size=0.33, random_state=42)



        # UFF [df["cost"][x] if (df["cost"][x] >= q1-1.5*iQR and df["cost"][x]<=q3+1.5*iQR) else q1 if (df["cost"][x] <= q1 - 1.5 * iQR) else q3 for x in good_ids]
        #[df.drop(x, inplace=True) for x in good_ids if df["cost"][x] <= q1 - 1.5 * iQR or df["cost"][x] >= q3 + 1.5 * iQR]
        #[df.drop(x, inplace=True) for x in good_ids if df["cost"][x] <= q1-1.5*iQR or df["cost"][x] >= q3+1.5*iQR]
        #df = df.reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["cost"]), df["cost"], test_size=0.33, random_state=42)

#RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

grid = {'n_estimators': [200,1000],
               'max_depth': [10, 100],
               'min_samples_split':[2, 10],
               'min_samples_leaf': [1, 4],
               'bootstrap': [True]}#}, False]}

#alessia = pd.read_excel("witholeh.xlsx")

rf = RandomForestRegressor(random_state = 42)#, n_estimators= 200, min_samples_split = 2, min_samples_leaf = 2, max_features ='auto', max_depth= 80, bootstrap= True)
#rf.fit(X_train, y_train)

#rf_random = rf
something = GridSearchCV(rf, param_grid=grid, cv=5, verbose=2)
something.fit(X_train, y_train)
something.best_params_

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

for x in range(100):
    new_ob = df.iloc[0]
    new_ob["mode_Truck"] = 0
    new_ob["mode_Air"] = 1
    yy = new_ob["cost"]
    new_ob = new_ob.drop("cost")
    predictions = something.predict([new_ob])
    errors = abs(predictions - yy)
    print(errors)

predictions
new_ob["mode_Truck"] = 1
new_ob["mode_Air"] = 0

df.to_excel("witholeh.xlsx")

#for classe in []:
##    for var in classe:
#        good_ids  =[x for x in range(len(df["weight"])) if int(df[var][x])==1]
#        olehh = pd.DataFrame([df["cost"][x] for x in good_ids]).describe()
        #df["cost"].astype(np.float64).describe()
#        q1, q3 = olehh.values[4], olehh.values[-2]
 #       iQR = q3 - q1
  #      [df.drop(x, inplace=True) for x in good_ids if df["cost"][x] <= q1-1.5*iQR or df["cost"][x] >= q3+1.5*iQR]
   #     df = df.reset_index(drop=True)

# PLACEHOLDERRRRRRRRRRRRRRRRRR

# good_ids  =[x for x in range(len(df["weight"])) if int(df["mode_Ocean"][x])==1]
# olehh = pd.DataFrame([df["cost"][x] for x in good_ids]).describe()
# q1, q3 = olehh.values[4], olehh.values[-2]
# [df.drop(x, inplace=True) for x in good_ids if df["cost"][x] <= q1 or df["cost"][x] >= q3]
# df = df.reset_index(drop=True)
#
# good_ids  =[x for x in range(len(df["weight"])) if int(df["mode_Air Charter"][x])==1]
# olehh = pd.DataFrame([df["cost"][x] for x in good_ids]).describe()
# q1, q3 = olehh.values[4], olehh.values[-2]
# [df.drop(x, inplace=True) for x in good_ids if df["cost"][x] <= q1 or df["cost"][x] >= q3]
# df = df.reset_index(drop=True)


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


import matplotlib.pyplot as plt
#mean square error -- non tuned vs tuned, 1 plot 2 lines
X_grid = [1,2,3,4,5]

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value

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
