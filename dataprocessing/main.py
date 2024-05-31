from dataprocessing.dataprocessing import remove_duplicate_player_ids, replace_dots_with_commas

if __name__ == '__main__':
    remove_duplicate_player_ids(file='playerIDs.csv')
    replace_dots_with_commas(file='merged_roster_20231217_0024.csv')
