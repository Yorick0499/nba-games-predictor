import pandas as pd
import re

data = pd.read_csv('data/TeamStatistics.csv')

data = data[['gameDate','teamCity','teamName','opponentTeamCity','opponentTeamName','win','assists','blocks',
             'steals', 'fieldGoalsPercentage','threePointersPercentage','freeThrowsPercentage',
             'reboundsDefensive','reboundsOffensive','reboundsTotal','foulsPersonal','turnovers','teamScore','opponentScore']]

data2 = data.loc[:,:]

data2['teamName'] = data2['teamCity']+' '+data2['teamName']
data2 = data2[data2['teamName']=='Utah Jazz'].head()
data2 = data2.sort_values(by='gameDate',ascending=True)
data2.drop('teamCity',axis=1,inplace=True)
data2['gameDate'] = pd.to_datetime(data2['gameDate'])
data2['gameDate'] = data2['gameDate'].dt.strftime('%Y-%m-%d')
data2['gameDate'] = pd.to_datetime(data2['gameDate'])
data2['diff']=data2.groupby('teamName')['gameDate'].diff().dt.days.fillna(7)
data2['daysOff'] = data2['diff'].astype('int')
data2.drop('diff',axis=1,inplace=True)
print(data2)


# print(data.head())
# print(data.dtypes)