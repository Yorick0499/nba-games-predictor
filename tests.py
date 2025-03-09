import pandas as pd
import pickle

# data = pd.read_csv('data/TeamStatistics.csv')

# data = data[['gameDate','teamCity','teamName','opponentTeamCity','opponentTeamName','win','assists','blocks',
#              'steals', 'fieldGoalsPercentage','threePointersPercentage','freeThrowsPercentage',
#              'reboundsDefensive','reboundsOffensive','reboundsTotal','foulsPersonal','turnovers','teamScore','opponentScore']]

# data2 = data.loc[:,:]

# data2['teamName'] = data2['teamCity']+' '+data2['teamName']
# data2 = data2[data2['teamName']=='Utah Jazz'].head()
# data2 = data2.sort_values(by='gameDate',ascending=True)
# data2.drop('teamCity',axis=1,inplace=True)
# data2['gameDate'] = pd.to_datetime(data2['gameDate'])
# data2['gameDate'] = data2['gameDate'].dt.strftime('%Y-%m-%d')
# data2['gameDate'] = pd.to_datetime(data2['gameDate'])
# data2['diff']=data2.groupby('teamName')['gameDate'].diff().dt.days.fillna(7)
# data2['daysOff'] = data2['diff'].astype('int')
# data2.drop('diff',axis=1,inplace=True)
# print(data2)


# # print(data.head())
# # print(data.dtypes)



# data = pd.read_csv('data/TeamStatistics.csv')
# data['gameDate'] = pd.to_datetime(data['gameDate'])
# data = data[data['gameDate']>='2020-10-22']

# dataCopy1 = data.copy()
# dataCopy2 = data.copy()

# print(data.dtypes)

# dataCopy1 = dataCopy1.iloc[:,[0,1,2,3,8,10,12,13,14,17,20,23,24,25,26,27,28]]
# dataCopy2 = dataCopy2.iloc[:,[0,5,6,8,11,12,13,14,17,20,23,24,25,26,27,28]]

# dataMerged = pd.merge(dataCopy1,dataCopy2,on='gameId')
# dataMerged = dataMerged.drop_duplicates(subset='gameId')

# dataMerged['teamName'] = dataMerged['teamCity']+' '+dataMerged['teamName']
# dataMerged.drop('teamCity',axis=1,inplace=True)
# dataMerged['opponentTeamName'] = dataMerged['opponentTeamCity']+' '+dataMerged['opponentTeamName']
# dataMerged.drop('opponentTeamCity',axis=1,inplace=True)


# dataMerged.to_csv('data/train.csv',index=False)





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
team = 'Sacramento Kings'
jazz_data = data2[data2['teamName']==f'{team}'].head(8)
print(jazz_data)

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


model = pickle.load(open('bin/XGBR_Model_v3.pkl','rb'))

print(model.predict(jazz_data))