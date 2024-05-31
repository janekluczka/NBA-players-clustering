import pandas as pd


def remove_duplicate_player_ids(file: str):
    file_df = pd.read_csv(file)
    # Remove duplicates from the 'PlayerID' column
    df_unique = file_df.drop_duplicates(subset='Player ID')
    # Save the unique values to a new CSV file
    df_unique.to_csv(f'unique_{file}', index=False)


def replace_dots_with_commas(file: str):
    file_df = pd.read_csv(file)
    # Replace all '.' with ',' while keeping NaN values unchanged
    file_df = file_df.applymap(lambda x: str(x).replace('.', ',') if pd.notna(x) else x)
    # Save the modified DataFrame to a new CSV file with ';' as separator
    file_df.to_csv(f'modified_{file}', index=False, sep=';')