import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np

data = pd.read_csv('data/TeamStatistics.csv')
print(data.dtypes)
data = data[['gameDate','teamCity','teamName','opponentTeamCity','opponentTeamName','win','assists','blocks',
             'steals','fieldGoalsAttempted','fieldGoalsMade', 'fieldGoalsPercentage','threePointersAttempted',
             'threePointersMade','threePointersPercentage','freeThrowsAttempted','freeThrowsMade','freeThrowsPercentage',
             'reboundsDefensive','reboundsOffensive','reboundsTotal','foulsPersonal','turnovers','teamScore','opponentScore']]
data.drop([0,1],axis=0,inplace=True)
data = data.reset_index(drop=True)

data['gameDate'] = pd.to_datetime(data['gameDate'])
data = data[data['gameDate']>='2020-10-22']
data['teamName'] = data['teamCity']+' '+data['teamName']
data['opponentTeamName'] = data['opponentTeamCity']+' '+data['opponentTeamName']

data.drop(['teamCity','opponentTeamCity'],axis=1,inplace=True)





# Tutaj zdefiniowac ostatnie 5 meczow dla druzyny, dla ktorej przewiduje sie wynik
team = 'Los Angeles Clippers'
jazz_data = data[data['teamName']==f'{team}'].head()
print(jazz_data)

data.drop(['gameDate','teamName','opponentTeamName','win'],axis=1,inplace=True)
# Tutaj wyrzucic to samo co z zestawu treningowego
jazz_data.drop(['gameDate','teamName','opponentTeamName','win'],axis=1,inplace=True)


# Zmiana kolejnosci, zeby pasowalo do danych wprowadzanych
jazz_sort = ['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']
jazz_data = jazz_data[jazz_sort]
jazz_data = jazz_data.mean()
jazz_data = list([jazz_data])
jazz_data = pd.DataFrame(jazz_data)







X = data.iloc[:,:17]
sort = ['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']
X = X[sort]
y = data.iloc[:,17:18]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)


model = RandomForestRegressor(n_estimators=100,random_state=123)
model.fit(X_train,y_train)


y_pred = model.predict(X_test)

print(root_mean_squared_error(y_test,y_pred))


print(f'{team}: {model.predict(jazz_data)}')








# print(model.predict(new))

# y_pred = model.predict(X_test_scaled)
# resid = y_test - y_pred
# print(X_train.dtypes)
# print(y_pred)
# print(y_test)
# print(resid)
# print(stats.shapiro(resid))
# print(mean_squared_error(y_test,y_pred))

# Wziac 5 ostatnich rzeczywistych mecz albo mniej (nieistotna liczba), ale obliczyc z tych ostatnich meczy srednia punktow i sprawdzic czy nie jes taka
# sama jak predict modelu ze srednich cech

# Jak regresja dostanie srednie statystyki (cechy), to zwroci sredni wynik i to jest normalna cecha regresji, wiec nie ma sensu ja dawac do tego zadania