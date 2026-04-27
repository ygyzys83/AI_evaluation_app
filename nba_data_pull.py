import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


def get_recent_nba_games_full(game_count=50):
    # '22025' is the code for the 2025-26 Regular Season
    # Use '42025' if you want only 2026 Playoff games
    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable='2025-26',
                                                    league_id_nullable='00')
    all_games = game_finder.get_data_frames()[0]

    # 1. Sort by date so we get the most recent games first
    all_games = all_games.sort_values(by='GAME_DATE', ascending=False)

    # 2. Get the unique Game IDs for the most recent X games
    recent_game_ids = all_games['GAME_ID'].unique()[:game_count]

    # 3. Filter the original dataframe to keep BOTH rows for those Game IDs
    full_game_data = all_games[all_games['GAME_ID'].isin(recent_game_ids)]

    # 4. Sort by Game ID and Team to keep them paired together nicely
    full_game_data = full_game_data.sort_values(by=['GAME_DATE', 'GAME_ID'], ascending=False)

    return full_game_data


if __name__ == "__main__":
    df = get_recent_nba_games_full(50)  # 50 games = 100 rows
    df.to_csv("nba_golden_dataset.csv", index=False)
    print(f"Successfully saved {len(df)} rows for {len(df) // 2} games to nba_golden_dataset.csv")