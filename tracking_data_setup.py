import pandas as pd
import numpy as np

def load_data():
    player_play = pd.read_csv('data/player_play.csv')
    plays = pd.read_csv('data/plays.csv')
    games = pd.read_csv('data/games.csv')
    player_play['causedPressure'] = player_play['causedPressure'].astype(int)
    pressure_on_play = player_play.groupby(['gameId', 'playId'])['causedPressure'].max().reset_index() #binary column to indicate whether there was at least one pressure on the play
    data = pd.merge(pressure_on_play, plays, on = ['gameId', 'playId'], how='left')
    model_data = data[(data['isDropback']) & (data['qbSpike'] != True)].copy() #only passing plays (plays where qb had a dropback)
    model_data = pd.merge(model_data, games.loc[:, ['gameId', 'homeTeamAbbr', 'visitorTeamAbbr', 'week']], on = 'gameId', how ='left')
    return model_data, plays

def create_tracking_data_features(model_data, plays):
    linespread = pd.DataFrame()
    maxdist = pd.DataFrame()
    gapinfluence = pd.DataFrame()
    line_prox = pd.DataFrame()
    for i in range(1, 10):
        tracking = pd.read_csv(f'data/tracking_week_{i}.csv')
        tracking = pd.merge(tracking, plays.loc[:, ['gameId', 'playId', 'absoluteYardlineNumber']], on = ['gameId', 'playId'], how='left') #add absoluteYardLine
        tracking_df = set_up_tracking(tracking, plays)
        linespread = pd.concat([linespread, create_linspread_feature(tracking_df)]) 
        maxdist = pd.concat([maxdist, create_max_distance_to_offense(tracking_df)])
        gapinfluence = pd.concat([gapinfluence, build_gap_influence(tracking_df, sigma=2.0)])
        line_prox = pd.concat([line_prox, line_proximity(tracking_df)])
    
    model_data = pd.merge(model_data, linespread, on = ['gameId', 'playId'], how = 'left')
    model_data = pd.merge(model_data, maxdist, on = ['gameId', 'playId'], how = 'left')
    model_data = pd.merge(model_data, gapinfluence , on = ['gameId', 'playId'], how = 'left')
    model_data = pd.merge(model_data, line_prox, on = ['gameId', 'playId'], how = 'left')
    return model_data

def set_up_tracking(tracking_data, plays):
    """
    Filters tracking data to last frame before the snap, adds player positions, and creates distance from line of scrimmage column
    """
    #Set up tracking data
    players = pd.read_csv('data/players.csv')
    tracking_pre_snap = tracking_data[tracking_data['frameType'] == "BEFORE_SNAP"]
    tracking_pre_snap = pd.merge(tracking_pre_snap, plays[['gameId', 'playId', 'isDropback']], on = ['gameId', 'playId'], how = 'inner')
    tracking_pre_snap = tracking_pre_snap[tracking_pre_snap['isDropback']]

    tracking_pre_snap = tracking_pre_snap.groupby(['gameId', 'playId', 'nflId']).last().reset_index()
    #add player positions to filter out defensive backs
    positions = players[['nflId', 'position']]
    tracking_pre_snap = pd.merge(tracking_pre_snap, positions, on = 'nflId', how='inner')
    #find vertical distance between player and football
    tracking_pre_snap['distance_from_line'] = np.abs(tracking_pre_snap['x'] - tracking_pre_snap['absoluteYardlineNumber'])

    #create binary column for offense or defense    
    position_dict = {
    'G':1, 'DT':0, 'C':1, 'DE':0, 'T':1, 'NT':0, 'OLB':0, 'MLB':0, 'ILB':0, 'LB':0, 'WR':1, 'CB':0, 'QB':1, 'SS':0, 'RB':1, 'TE':1, 'FS':0, 'FB':1, 'DB':0
    }
    tracking_pre_snap['isOffense'] = tracking_pre_snap['position'].map(position_dict)
    return tracking_pre_snap

def create_linspread_feature(tracking_pre_snap):
    """
    Takes tracking data and calculates the spread among defensive lineman (max y - min y of defensive lineman) before the snap and returns this as a feature in the overall model_data
    """
    tracking_pre_snap = tracking_pre_snap[tracking_pre_snap['position'].isin(['DE', 'NT', 'SS', 'FS', 'OLB', 'DT', 'CB', 'ILB', 'MLB', 'DB', 'LB'])]
    #filter out DBs and have other players need to be within 1.5 yards of the ball to be determined on the line
    line_spread = tracking_pre_snap.loc[(tracking_pre_snap['distance_from_line'] <= 2) & ~(tracking_pre_snap['position'].isin(['SS', 'FS', 'DB', 'CB']))]
    #find minimum horziontal and maximum horizontal position of filtered lineman
    spread = line_spread.groupby(['gameId', 'playId'])['y'].agg(['max', 'min']).reset_index()
    #calculate dline spread
    spread['dlinespread'] = spread['max'] - spread['min']
    return spread.loc[:,['gameId', 'playId', 'dlinespread']]

def create_max_distance_to_offense(tracking_df):
    """
    Calculates minimum distance to an offensive lineman for each defensive lineman, and takes the max of these distances for each play and returns it as a feature in model_data
    """
    lineman = tracking_df.loc[(tracking_df['distance_from_line'] <= 2) & (tracking_df['position'].isin(['DE', 'NT', 'OLB', 'DT', 'ILB', 'MLB', 'LB', 'G', 'C', 'T']))].copy()
    lineman.loc[:, 'max_distance_to_offense'] = np.nan
    for (game, play), play_df in lineman.groupby(['gameId', 'playId']):
        offense = play_df[play_df['isOffense'] == 1]
        defense = play_df[play_df['isOffense'] == 0]

        offense_coords = offense[['x', 'y']].values
        defense_coords = defense[['x', 'y']].values

        distances = np.sqrt(
                ((defense_coords[:, np.newaxis, 0] - offense_coords[:, 0]) ** 2) +
                ((defense_coords[:, np.newaxis, 1] - offense_coords[:, 1]) ** 2)
            )
        
        if distances.size > 0: min_distances = distances.min(axis=1)
        else: min_distances = None

        lineman.loc[defense.index, 'max_distance_to_offense'] = min_distances

    max_lineman_dist = lineman.groupby(['gameId', 'playId'])['max_distance_to_offense'].max().reset_index()
    return max_lineman_dist

def find_gap_coords(tracking_df):
    olineman = tracking_df.loc[(tracking_df['position'].isin(['DE', 'NT', 'OLB', 'DT', 'ILB', 'MLB', 'LB', 'G', 'C', 'T']) & (tracking_df['isOffense'] == 1))].copy()
    results = []
    for (gameId, playId), group in olineman.groupby(['gameId', 'playId']):
        coords = group.sort_values('y')[['x', 'y']]
        gaps = [(coords.iloc[0, 0], coords.iloc[0, 1] - 1.5)]
        for i in range(len(coords)-1):
            gap_x = (coords.iloc[i, 0] + coords.iloc[i+1, 0]) / 2
            gap_y = (coords.iloc[i, 1] + coords.iloc[i+1, 1]) / 2
            gaps.append((gap_x, gap_y))
        gaps.append((coords.iloc[4, 0], coords.iloc[4, 1] + 1.5))
        results.append({'gameId':gameId, 'playId':playId, 'gaps':gaps})
    gaps_df = pd.DataFrame(results)
    return gaps_df

def find_dline_coords(tracking_df):
    dlineman = tracking_df.loc[(tracking_df['distance_from_line'] <= 2) & (tracking_df['position'].isin(['DE', 'NT', 'OLB', 'DT', 'ILB', 'MLB', 'LB']))].copy()
    results = []
    for (gameId, playId), group in dlineman.groupby(['gameId', 'playId']):
        coords = group.sort_values('y')
        coords = list(zip(coords['x'], coords['y']))
        results.append({'gameId':gameId, 'playId':playId, 'dline_coords':coords})
    dline_coords_df = pd.DataFrame(results)
    return dline_coords_df

def build_gap_influence(tracking_df, sigma):
    dline_coords_df = find_dline_coords(tracking_df)
    gaps_df = find_gap_coords(tracking_df)
    dline_influence_df = pd.merge(dline_coords_df, gaps_df, on = ['gameId', 'playId'], how = 'left')

    gap_influence = pd.DataFrame(columns=['gameId', 'playId', 'gap1_influence', 'gap2_influence', 'gap3_influence', 'gap4_influence', 'gap5_influence', 'gap6_influence'])
    for (gameId, playId), group in dline_influence_df.groupby(['gameId', 'playId']):
        d_line = np.array(group['dline_coords'].tolist()).squeeze()
        gaps = np.array(group['gaps'].tolist()).squeeze()
        n_players = len(d_line)
        n_gaps = len(gaps)
        influences = np.zeros((n_gaps, n_players))

        for i, player_pos in enumerate(d_line):
            for j, gap_pos in enumerate(gaps):
                distance = np.sum((player_pos - gap_pos)**2)
                influences[j, i] =  np.exp(-distance / (2*sigma**2))
        
        player_columns = [f'Player_{i+1}' for i in range(n_players)]
        df_influence = pd.DataFrame(influences, columns=player_columns)
        
        # Add gap coordinates and total influence
        df_influence['Gap_X'] = gaps[:, 0]
        df_influence['Gap_Y'] = gaps[:, 1]
        df_influence['Total_Influence'] = df_influence[player_columns].sum(axis=1)
        if n_gaps == 6:
            gap_influence.loc[len(gap_influence)] = [gameId, playId] + df_influence[player_columns].sum(axis=1).tolist()
        else:
            gap_influence.loc[len(gap_influence)] = [gameId, playId] + df_influence[player_columns].sum(axis=1).tolist()[:6]
    return gap_influence

def line_proximity(tracking_df):
    line_proximity = tracking_df.groupby(['gameId', 'playId'])['distance_from_line'].agg({'max', 'mean'}).reset_index()
    line_proximity = line_proximity.rename({'mean':'avg_def_dist_from_line', 'max':'max_def_dist_from_line'}, axis=1)
    return line_proximity


def main():
    model_data, plays = load_data()
    model_data = create_tracking_data_features(model_data, plays)
    model_data.to_csv('model_data.csv')
if __name__ == "__main__":
    main()