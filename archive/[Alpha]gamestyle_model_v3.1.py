import pandas as pd
from scipy import stats
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np

# To są dane treningowe do czyszczenia
data = pd.read_csv('data/train.csv')
data.drop([0],axis=0,inplace=True)
data = data.reset_index(drop=True)
data.drop(['gameId','home_y'],inplace=True,axis=1)
data = data.rename(columns={'opponentScore':'teamScore_y',
                            'teamScore':'teamScore_x'})


# To są najświeższe dane o meczach
data2 = pd.read_csv('data/LatestData.csv')
data2 = data2[['gameDate','teamCity','teamName','opponentTeamCity','opponentTeamName','win','assists','blocks',
             'steals', 'fieldGoalsPercentage','threePointersPercentage','freeThrowsPercentage',
             'reboundsDefensive','reboundsOffensive','reboundsTotal','foulsPersonal','turnovers','teamScore','opponentScore']]
data2.drop([0,1],axis=0,inplace=True)
data2 = data2.reset_index(drop=True)
data2['gameDate'] = pd.to_datetime(data2['gameDate'])
data2 = data2[data2['gameDate']>='2020-10-22']
data2['teamName'] = data2['teamCity']+' '+data2['teamName']
data2['opponentTeamName'] = data2['opponentTeamCity']+' '+data2['opponentTeamName']
data2.drop(['teamCity','opponentTeamCity'],axis=1,inplace=True)





# Tutaj wyszukac ostatnie 5 meczow dla druzyny, dla ktorej przewiduje sie wynik
team = 'Toronto Raptors'
jazz_data = data2[data2['teamName']==f'{team}'].head(8)
print(jazz_data)

# Wyrzucic niepotrzebne kolumny ze zbioru, z ktorego bedzie sie uczyl
data.drop(['gameDate','teamName','opponentTeamName'],axis=1,inplace=True)

# Tutaj wyrzucic to samo co z zestawu treningowego
jazz_data.drop(['gameDate','teamName','opponentTeamName','win'],axis=1,inplace=True)

# Zmiana kolejnosci, zeby pasowalo do danych wprowadzanych
jazz_sort = ['fieldGoalsPercentage','threePointersPercentage','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']
jazz_data = jazz_data[jazz_sort]
jazz_data = jazz_data.mean()
jazz_data = list([jazz_data])
jazz_data = pd.DataFrame(jazz_data)

opTeam = 'Washington Wizards'
jazz_data2 = data2[data2['teamName']==f'{opTeam}'].head(8)
print(jazz_data2)
jazz_data2.drop(['gameDate','teamName','opponentTeamName','win'],axis=1,inplace=True)
jazz_data2 = jazz_data2[jazz_sort]
jazz_data2 = jazz_data2.mean()
jazz_data2 = list([jazz_data2])
jazz_data2 = pd.DataFrame(jazz_data2)

jazz_data_concated = pd.concat([jazz_data,jazz_data2],axis=1)
jazz_data_concated['home'] = 0



# Przesortowanie data
sort = ['fieldGoalsPercentage_x','threePointersPercentage_x','freeThrowsPercentage_x',
        'reboundsOffensive_x','reboundsDefensive_x','reboundsTotal_x',
        'assists_x','steals_x','blocks_x','turnovers_x','foulsPersonal_x',
        'fieldGoalsPercentage_y','threePointersPercentage_y','freeThrowsPercentage_y',
        'reboundsOffensive_y','reboundsDefensive_y','reboundsTotal_y',
        'assists_y','steals_y','blocks_y','turnovers_y','foulsPersonal_y','home_x','teamScore_x','teamScore_y']
data = data[sort]







# Trenowanie modelu
X = data.iloc[:,:23]
nan_index = X[X.isna().any(axis=1)].index
X = X.drop(nan_index)
y = data.iloc[:,23:24]
y = y.drop(nan_index)
y = y.squeeze()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)
model = XGBRegressor(n_estimators=300)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

scores = cross_val_score(model,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
print(f'Cross-Val RMSE - zbiór treningowy: {np.sqrt(-scores.mean())}')
print(f'RMSE - zbiór testowy: {root_mean_squared_error(y_test,y_pred)}')
print(f'Model score: {model.score(X_test,y_test)}')
print('\nWynik:')
print(f'{team}: {model.predict(np.array(jazz_data_concated))}')








