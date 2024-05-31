import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from scrapper.scrapper import get_team_players_data


def scrap_team_players_data(
        teams: list[str],
        start_season_end_year: int = 2023,
        last_season_end_year: int = 2023
):
    # Remove duplicates and sort alphabetically
    unique_team_short_names: set[str] = set(sorted(set(teams)))

    # Create an empty list to store individual rosters
    rosters = []

    # Get player stats for each team in unique_team_short_names in range from start_season_end_year to last_season_end_year
    for year in range(start_season_end_year, last_season_end_year + 1):
        for selected_team in tqdm(unique_team_short_names, desc=f"Year {year}", unit="team"):
            roster = get_team_players_data(team=selected_team, season_end_year=year)

            if roster is not None:
                rosters.append(roster)

            # Wait 3.0 seconds to ensure compliance with the 20 requests/minute policy
            time.sleep(3.0)

    # Merge all rosters into a single DataFrame
    merged_roster = pd.concat(rosters, ignore_index=True)

    # Get the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    # Save the merged DataFrame to a CSV file with the current time in the filename
    csv_filename = f'merged_roster_{current_time}.csv'
    merged_roster.to_csv(csv_filename, index=False)
    print(f"Merged roster saved to {csv_filename}")


if __name__ == '__main__':
    team_short_names = [
        'ATL', 'SLH', 'MIL', 'TCB', 'BOS', 'BRK', 'NJN', 'CHI', 'CHH', 'CHO',
        'CHA', 'CLE', 'DAL', 'DEN', 'DET', 'FWP', 'GSW', 'SFW', 'PHI', 'HOU',
        'IND', 'LAC', 'SDC', 'BUF', 'LAL', 'MIN', 'MEM', 'VAN', 'MIA', 'MIL',
        'MIN', 'NOP', 'NOK', 'NOH', 'NYK', 'OKC', 'SEA', 'ORL', 'PHI', 'SYR',
        'PHO', 'POR', 'SAC', 'KCK', 'KCK', 'CIN', 'ROR', 'SAS', 'TOR', 'UTA',
        'NOJ', 'WAS', 'WAS', 'CAP', 'BAL', 'CHI', 'CHI', 'AND', 'CHI', 'IND',
        'SRS', 'SLB', 'WAS', 'WAT', 'SDR'
    ]

    scrap_team_players_data(team_short_names)
