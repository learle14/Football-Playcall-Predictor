import nfl_data_py as nfl

# grab multiple seasons in one call — pass a list of years
years = list(range(2015, 2024))
pbp = nfl.import_pbp_data(years)

print(pbp.shape)
pbp.head()

df = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
df = df[df['down'].notna()]  # drop plays without a down (e.g. some special teams)

cols = ['play_type', 'down', 'ydstogo', 'yardline_100', 
        'score_differential', 'game_seconds_remaining', 
        'posteam', 'defteam', 'qtr', 'shotgun', 'no_huddle']
df = df[cols]