import pandas as pd

data = pd.read_csv('data/LatestData.csv')
print(data[(data['teamName']=='Warriors')&(data['opponentTeamName']=='Knicks')].loc[:,['teamCity','teamName','win','opponentTeamName','opponentTeamCity','gameDate']].head())
print(data[data['teamName']=='Warriors'].loc[:,['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']].head().mean())
print()
print(data[data['teamName']=='Knicks'].loc[:,['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']].head().mean())