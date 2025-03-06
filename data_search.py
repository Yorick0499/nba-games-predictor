import pandas as pd

data = pd.read_csv('data/LatestData.csv')
print(data[(data['teamName']=='Celtics')&(data['opponentTeamName']=='Trail Blazers')].loc[:,['teamCity','teamName','win','opponentTeamName','opponentTeamCity','teamScore','opponentScore','gameDate']].head())
print(data[data['teamName']=='Celtics'].loc[:,['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']].head().mean())
print()
print(data[data['teamName']=='Trail Blazers'].loc[:,['fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage',
        'threePointersMade','threePointersAttempted','threePointersPercentage',
        'freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage',
        'reboundsOffensive','reboundsDefensive','reboundsTotal',
        'assists','steals','blocks','turnovers','foulsPersonal']].head().mean())
print()
print(data[data['teamName']=='Celtics'].loc[:,['teamName','win','opponentTeamName','teamScore','opponentScore']].head(10))
print()
print(data[data['teamName']=='Trail Blazers'].loc[:,['teamName','win','opponentTeamName','teamScore','opponentScore']].head(10))