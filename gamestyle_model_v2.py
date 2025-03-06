import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np

# To są dane treningowe do czyszczenia
data = pd.read_csv('data/TeamStatistics.csv')
data = data[['gameDate','teamCity','teamName','opponentTeamCity','opponentTeamName','home','win','assists','blocks',
             'steals', 'fieldGoalsPercentage','threePointersPercentage','freeThrowsPercentage',
             'reboundsDefensive','reboundsOffensive','reboundsTotal','foulsPersonal','turnovers','teamScore','opponentScore']]
data.drop([0,1],axis=0,inplace=True)
data = data.reset_index(drop=True)
data['gameDate'] = pd.to_datetime(data['gameDate'])
data = data[data['gameDate']>='2020-10-22']
data['teamName'] = data['teamCity']+' '+data['teamName']
data['opponentTeamName'] = data['opponentTeamCity']+' '+data['opponentTeamName']
data.drop(['teamCity','opponentTeamCity'],axis=1,inplace=True)

# Dodanie dodatkowej cechy daysOff - dni przerwy miedzy meczami
data = data.sort_values(by='gameDate',ascending=True)
data['gameDate'] = data['gameDate'].dt.strftime('%Y-%m-%d')
data['gameDate'] = pd.to_datetime(data['gameDate'])
data['daysOff'] = data.groupby('teamName')['gameDate'].diff().dt.days.fillna(7)
data['daysOff'] = data['daysOff'].astype('int')


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
team = 'Denver Nuggets'
jazz_data = data2[data2['teamName']==f'{team}'].head(8)
print(jazz_data)

# Wyrzucic niepotrzebne kolumny ze zbioru, z ktorego bedzie sie uczyl
data.drop(['gameDate','teamName','opponentTeamName','win'],axis=1,inplace=True)

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

# Tutaj nalezy okreslic, ile bylo dni przerwy oraz czy mecz bedzie domowy czy nie
jazz_data['daysOff'] = 2
jazz_data['home'] = 0
print(jazz_data)



# Przesortowanie data
sort = ['fieldGoalsPercentage','threePointersPercentage','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal','daysOff','home','teamScore','opponentScore']
data = data[sort]





# Trenowanie modelu
X = data.iloc[:,:13]
nan_index = X[X.isna().any(axis=1)].index
X = X.drop(nan_index)
y = data.iloc[:,13:14]
y = y.drop(nan_index)
y = y.squeeze()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)
model = RandomForestRegressor(n_estimators=100,random_state=123)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(root_mean_squared_error(y_test,y_pred))
print(model.score(X_test,y_test))
print(f'{team}: {model.predict(jazz_data)}')








