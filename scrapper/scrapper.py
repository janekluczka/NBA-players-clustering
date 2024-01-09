import pandas as pd
from bs4 import BeautifulSoup, Comment
from requests import get


def get_player_awards():
    # Fetch the HTML content from the website
    response = get('https://www.basketball-reference.com/awards/smoy.html')

    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        print(f"Failed to retrieve SMOY data. Status code: {response.status_code}")
        return None

    # Parse HTML content
    soup_html = BeautifulSoup(response.content, 'html.parser')

    # TODO: Read team stats
    tables = soup_html.find_all('table')

    smoy_NBA_table = soup_html.select_one('table#smoy_NBA')
    if smoy_NBA_table:
        smoy_NBA_dataframe = pd.read_html(str(smoy_NBA_table))[0]

    pass


def get_team_players_data(team: str, season_end_year: int, has_advanced: bool = False, has_shooting: bool = False):
    """
    Retrieve and process basketball team data from basketball-reference.com.

    Parameters:
    - team (str): The abbreviation of the basketball team (e.g., 'LAL' for Los Angeles Lakers).
    - season_end_year (int): The ending year of the basketball season (e.g., 2022 for the 2021-2022 season).
    - has_advanced (bool)
    -

    Returns:
    - pd.DataFrame: Merged DataFrame containing per-game, advanced, and shooting statistics.
    """

    # Fetch the HTML content from the website
    response = get(f'https://www.basketball-reference.com/teams/{team}/{season_end_year}.html')

    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        print(f"Failed to retrieve data for {team}. Status code: {response.status_code}")
        return None

    # Parse HTML content
    soup_html = BeautifulSoup(response.content, 'html.parser')
    soup_lxml = BeautifulSoup(response.content, 'lxml')
    soup_lxml_no_comments = BeautifulSoup("\n".join(soup_lxml.find_all(text=Comment)), "lxml")

    tables_html = soup_html.find_all('table')
    tables_lxml = soup_lxml_no_comments.find_all('table')

    team_misc_table = soup_lxml_no_comments.select_one('table#team_misc')
    per_game_table = soup_html.select_one('table#per_game')
    totals_table = soup_html.select_one('table#totals')

    if team_misc_table is None:
        print(f"Team Misc table not found for {team}")
        return None

    if per_game_table is None:
        print(f"Per game table not found for {team}")
        return None

    if totals_table is None:
        print(f"Totals table not found for {team}")
        return None

    player_df = create_player_df(per_game_table, season_end_year, team)
    team_misc_df = create_team_misc_df(team_misc_table)
    per_game_df = create_per_game_df(per_game_table)
    totals_df = create_totals_df(totals_table)

    # Merge the four DataFrames
    merged_dataframe = pd.merge(
        player_df,
        per_game_df,
        on=['Player'],
        how='outer'
    )

    merged_dataframe = pd.merge(
        merged_dataframe,
        totals_df,
        on=['Player', 'Age', 'G', 'GS', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'],
        how='outer'
    )

    for col in team_misc_df.columns:
        if col != 'Team':
            merged_dataframe[f'Team {col}'] = team_misc_df[col].iloc[0]

    if has_advanced:
        advanced_table = soup_html.select_one('table#advanced')
        advanced_dataframe = create_advanced_df(advanced_table)
        if advanced_dataframe is not None:
            merged_dataframe = pd.merge(
                merged_dataframe,
                advanced_dataframe,
                on=['Player', 'Age', 'G'] if not advanced_dataframe.empty else ['Player'],
                how='outer'
            )
        else:
            print(f"Advanced table not found for {team}")

    if has_shooting:
        shooting_table = soup_lxml_no_comments.select_one('table#shooting')
        shooting_dataframe = create_shooting_df(shooting_table)
        if shooting_dataframe is not None:
            merged_dataframe = pd.merge(
                merged_dataframe,
                shooting_dataframe,
                on=['Player', 'Age', 'G', 'MP', 'FG%'] if not shooting_dataframe.empty else ['Player'],
                how='outer'
            )
        else:
            print(f"Shooting table not found for {team}")

    # Drop columns with 'Rk' in their names (case-insensitive)
    merged_dataframe = merged_dataframe.loc[:, ~merged_dataframe.columns.str.contains('Rk', case=False)]
    # Drop columns with 'Unnamed' in their names (case-insensitive)
    merged_dataframe = merged_dataframe.loc[:, ~merged_dataframe.columns.str.contains('Unnamed', case=False)]

    print(f"Retrieved data for {team}")

    return merged_dataframe


def create_shooting_df(shooting_table):
    if shooting_table:
        shooting_dataframe = pd.read_html(str(shooting_table))[0]
        flattened_columns = []
        for column in shooting_dataframe.columns:
            if column[0].startswith('Unnamed') and column[1].startswith('Unnamed'):
                new_col_name = "Unnamed"
            elif column[0].startswith('Unnamed'):
                new_col_name = column[1]
            elif column[1].startswith('Unnamed'):
                new_col_name = column[0]
            else:
                new_col_name = f"{column[0]} {column[1]}"
            flattened_columns.append(new_col_name)
        shooting_dataframe.columns = flattened_columns
    else:
        shooting_dataframe = None
    return shooting_dataframe


def create_advanced_df(advanced_table):
    if advanced_table:
        advanced_dataframe = pd.read_html(str(advanced_table))[0]
    else:
        advanced_dataframe = None
    return advanced_dataframe


def create_per_game_df(per_game_table):
    per_game_df = pd.read_html(str(per_game_table))[0]
    columns_to_rename = [
        'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
        'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
    ]
    for column in columns_to_rename:
        per_game_df.rename(columns={column: f'{column} per game'}, inplace=True)
    return per_game_df


def create_totals_df(totals_table):
    totals_df = pd.read_html(str(totals_table))[0]
    columns_to_rename = [
        'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
        'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
    ]
    for column in columns_to_rename:
        totals_df.rename(columns={column: f'{column} Total'}, inplace=True)
    return totals_df


def create_team_misc_df(team_misc_table):
    team_misc_df = pd.read_html(str(team_misc_table))[0]
    flattened_columns = []
    for column in team_misc_df.columns:
        if column[0].startswith('Unnamed') and column[1].startswith('Unnamed'):
            new_col_name = "Unnamed"
        elif column[0].startswith('Unnamed'):
            new_col_name = column[1]
        elif column[1].startswith('Unnamed'):
            new_col_name = column[0]
        else:
            new_col_name = f"{column[0]} {column[1]}"
        flattened_columns.append(new_col_name)
    team_misc_df.columns = flattened_columns
    filtered_team_misc_dataframe = team_misc_df[team_misc_df.iloc[:, 0] == 'Team']
    # print(filtered_team_misc_dataframe.to_markdown())
    return filtered_team_misc_dataframe


def create_player_df(per_game_table, season_end_year, team):
    player_data = []
    for row in per_game_table.find_all('tr')[1:]:
        player_tag = row.find('td', {'data-stat': 'player'})
        player_id = player_tag.find('a')['href'].split('/')[-1].split('.')[0]
        player = player_tag.text.strip()
        player_data.append({
            'Player ID': player_id,
            'Player': player,
            "Team": team,
            "Season": f'{season_end_year - 1}/{season_end_year}'
        })
    player_dataframe = pd.DataFrame(player_data)
    return player_dataframe
