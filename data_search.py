import pandas as pd

data = pd.read_csv('data/LatestData.csv')
print(data[(data['teamName']=='Suns')&(data['opponentTeamName']=='Mavericks')].loc[:,['teamCity','teamName','win','opponentTeamName','opponentTeamCity','teamScore','opponentScore','gameDate']].head())
print(data[data['teamName']=='Suns'].loc[:,['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']].head().mean())
print()
print(data[data['teamName']=='Mavericks'].loc[:,['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']].head().mean())
print()
print(data[data['teamName']=='Suns'].loc[:,['teamName','win','opponentTeamName','teamScore','opponentScore']].head(10))
print()
print(data[data['teamName']=='Mavericks'].loc[:,['teamName','win','opponentTeamName','teamScore','opponentScore']].head(10))