
# coding: utf-8

# In[160]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[161]:


#load data 
world_cup = pd.read_csv('datasets/World Cup 2018 Dataset.csv')
results = pd.read_csv('datasets/results.csv')


# In[162]:


results.head()


# In[163]:


results.head()


# In[164]:


#Adding goal difference and establishing who is the winner 
winner_team = []
for i in range (len(results['home_team'])):
    if results ['home_score'][i] > results['away_score'][i]:
        winner_team.append(results['home_team'][i])
    elif results['home_score'][i] < results ['away_score'][i]:
        winner_team.append(results['away_team'][i])
    else:
        winner_team.append('Draw')
results['winning_team'] = winner_team

#adding goal difference column
results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])

results.head()


# In[165]:


#narrowing to team patcipating in the world cup
worldcup_teams = ['Australia', ' Iran', 'Japan', 'Korea Republic', 
            'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria', 
            'Senegal', 'Tunisia', 'Costa Rica', 'Mexico', 
            'Panama', 'Argentina', 'Brazil', 'Colombia', 
            'Peru', 'Uruguay', 'Belgium', 'Croatia', 
            'Denmark', 'England', 'France', 'Germany', 
            'Iceland', 'Poland', 'Portugal', 'Russia', 
            'Serbia', 'Spain', 'Sweden', 'Switzerland']
df_teams_home = results[results['home_team'].isin(worldcup_teams)]
df_teams_away = results[results['away_team'].isin(worldcup_teams)]
df_teams = pd.concat((df_teams_home, df_teams_away))
df_teams.drop_duplicates()
df_teams.count()


# In[166]:


df_teams.head()


# In[167]:


#create an year column to drop games before 1930
year = []
for row in df_teams['date']:
    year.append(int(row[:4]))
df_teams['match_year'] = year
df_teams.head()


# In[168]:


#dropping columns that wll not affect matchoutcomes
df_teams_req_cols = df_teams.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'goal_difference', 'match_year'], axis=1)
df_teams_req_cols.head()


# In[169]:


#Building the model
#categorize wining team to numerical category if home team wins : 2, draw : 1, away team wins : 0

df_teams_req_cols = df_teams_req_cols.reset_index(drop=True)
df_teams_req_cols.loc[df_teams_req_cols.winning_team == df_teams_req_cols.home_team,'winning_team']=2
df_teams_req_cols.loc[df_teams_req_cols.winning_team == 'Draw', 'winning_team']=1
df_teams_req_cols.loc[df_teams_req_cols.winning_team == df_teams_req_cols.away_team, 'winning_team']=0

df_teams_req_cols.head()


# In[170]:


#convert teams from categorical variables to continous inputs 

final = pd.get_dummies(df_teams_req_cols, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# Separate X and y sets
X = final.drop(['winning_team'], axis=1)
y = final["winning_team"]
y = y.astype('int')

# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)


# In[171]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))


# In[172]:


#if any of team is having higher fifa ranking will be considered as home team, else away team

# Loading new datasets
fifa_rankings = pd.read_csv('datasets/fifa_rankings.csv') 
fixtures = pd.read_csv('datasets/fixtures.csv')

pred_set = []


# In[173]:



# Creating ranking column for each team
fixtures.insert(1, 'first_ranking', fixtures['Home Team'].map(fifa_rankings.set_index('Team')['Position']))
fixtures.insert(2, 'second_ranking', fixtures['Away Team'].map(fifa_rankings.set_index('Team')['Position']))


fixtures.head()


# In[174]:


#based on ranking of each team, mark it as home team and away team
for index, row in fixtures.iterrows():
    if row['first_ranking'] < row['second_ranking']:
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None})
    else:
        pred_set.append({'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winning_team': None})
        
pred_set = pd.DataFrame(pred_set)
duplicate_pred_set = pred_set

pred_set.head()


# In[175]:


# convert home and away team to continous variable
pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# Add missing columns compared to the model's training dataset
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

# dropping winning team column
pred_set = pred_set.drop(['winning_team'], axis=1)

pred_set.head()


# In[176]:


#group matches 
predictions = logreg.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(duplicate_pred_set.iloc[i, 1] + " and " + duplicate_pred_set.iloc[i, 0])
    if predictions[i] == 2:
        print("Winner: " + duplicate_pred_set.iloc[i, 1])
    elif predictions[i] == 1:
        print("Draw")
    elif predictions[i] == 0:
        print("Winner: " + duplicate_pred_set.iloc[i, 0])
    print('Probability of ' + duplicate_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
    print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1]))
    print('Probability of ' + duplicate_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
    print("")


# In[177]:


# List of tuples before 
teams_16 = [('Uruguay', 'Portugal'),
            ('France', 'Croatia'),
            ('Brazil', 'Mexico'),
            ('England', 'Colombia'),
            ('Spain', 'Russia'),
            ('Argentina', 'Peru'),
            ('Germany', 'Switzerland'),
            ('Poland', 'Belgium')]


# In[178]:



def  clean_and_predict (matches, ranking, final, logreg):

    # Initialization of auxiliary list for data cleaning
    positions = []

    # Loop to retrieve each team's position according to FIFA ranking
    for match in matches:
        positions.append(ranking.loc[ranking['Team'] == match[0],'Position'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == match[1],'Position'].iloc[0])
    
    # Creating the DataFrame for prediction
    pred_set = []

    # Initializing iterators for while loop
    i = 0
    j = 0

    # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
    while i < len(positions):
        dict1 = {}

        # If position of first team is better, he will be the 'home' team, and vice-versa
        if positions[i] < positions[i + 1]:
            dict1.update({'home_team': matches[j][0], 'away_team': matches[j][1]})
        else:
            dict1.update({'home_team': matches[j][1], 'away_team': matches[j][0]})

        # Append updated dictionary to the list, that will later be converted into a DataFrame
        pred_set.append(dict1)
        i += 2
        j += 1

    # Convert list into DataFrame
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    # Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

    # Add missing columns compared to the model's training dataset
    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    # Remove winning team column
    pred_set = pred_set.drop(['winning_team'], axis=1)

    # Predict!
    predictions = logreg.predict(pred_set)
    for i in range(len(pred_set)):
        print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
        if predictions[i] == 2:
            print("Winner: " + backup_pred_set.iloc[i, 1])
        elif predictions[i] == 1:
            print("Draw")
        elif predictions[i] == 0:
            print("Winner: " + backup_pred_set.iloc[i, 0])
        print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ' , '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
        print('Probability of Draw: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1])) 
        print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
        print("")


# In[179]:


clean_and_predict(teams_16, fifa_rankings, final, logreg)


# In[180]:


# List of matches
quarters = [('Portugal', 'France'),
            ('Spain', 'Argentina'),
            ('Brazil', 'England'),
            ('Germany', 'Belgium')]


# In[181]:



clean_and_predict (quarters, fifa_rankings, final, logreg)


# In[182]:


# List of matches
semi = [('Portugal', 'Brazil'),
        ('Argentina', 'Germany')]


# In[183]:



clean_and_predict (semi, fifa_rankings, final, logreg)


# In[184]:



# Finals# Finals
finals = [('Brazil', 'Germany')]


# In[185]:


clean_and_predict(finals, fifa_rankings, final, logreg)


# In[186]:


top_16 = []
top_16_players = pd.DataFrame()
for team in teams_16:
    top_16.append(team[0])
    top_16.append(team[1])
top_16_players['16 players'] = top_16
top_16_players.head()


# In[187]:


top_quarters = []
top_quarters_players = pd.DataFrame()
for team in quarters:
    top_quarters.append(team[0])
    top_quarters.append(team[1])
top_quarters_players['quarterfinal'] = top_quarters
top_quarters_players.head()


# In[188]:


top_semi = []
top_semi_players = pd.DataFrame()
for team in semi:
    top_semi.append(team[0])
    top_semi.append(team[1])
top_semi_players['semifinal'] = top_semi
top_semi_players.head()


# In[189]:


top_finals = []
top_finals_players = pd.DataFrame()
for team in finals:
    top_finals.append(team[0])
    top_finals.append(team[1])
top_finals_players['final'] = top_finals
top_finals_players.head()


# In[190]:


winner_team=['Brazil']
winner_df = pd.DataFrame()
winner_df['winner'] = winner_team
winner_df.head(10)


# In[191]:


del fifa_result
fifa_result=pd.DataFrame()

fifa_result = pd.concat([top_16_players,top_quarters_players,top_semi_players,top_finals_players,winner_df],axis = 1)
fifa_result.head(16)


# In[193]:


fifa_result.to_csv('Football_cup_submission.csv.', sep=',',na_rep='',index=False)

