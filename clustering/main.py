# # Feature selection
# features = [
#     'Team W', 'Team L',
#     'MP per game', 'MP Total',
#     'PTS per game', 'PTS Total',
#     'PER', 'TS%', 'USG%',
#     'FG per game', 'FGA per game', 'FG%', 'FG Total', 'FGA Total', 'eFG%',
#     '2P per game', '2PA per game', '2P%', '2P Total', '2PA Total',
#     '3P per game', '3PA per game', '3P%', '3P Total', '3PA Total',
#     'FT per game', 'FTA per game', 'FT%', 'FT Total', 'FTA Total',
#     'ORB per game', 'ORB Total', 'ORB%',
#     'DRB per game', 'DRB Total', 'DRB%',
#     'TRB per game', 'TRB Total', 'TRB%',
#     'AST per game', 'AST Total', 'AST%',
#     'STL per game', 'STL Total', 'STL%',
#     'BLK per game', 'BLK Total', 'BLK%',
#     'TOV per game', 'TOV Total', 'TOV%',
#     'PF per game', 'PF Total',
#     'OWS', 'DWS', 'WS', 'WS/48',
#     'OBPM', 'DBPM', 'BPM',
#     'VORP'
# ]
import pandas as pd

from clustering.clustering import filter_data, select_features, kmeans_clustering, hierarchical_clustering, \
    gmm_clustering, spectral_clustering, plot_cluster_stats, plot_3d_cluster

# Feature selection
features = [
    # 'Team W',
    # 'Team L',
    'MP per game',
    'PTS per game',
    # 'eFG%',
    # 'TS%',
    # 'USG%',
    'FG per game', 'FG%',
    '2P per game', '2P%',
    '3P per game', '3P%',
    'FT per game', 'FT%',
    'ORB per game', 'ORB%',
    'DRB per game', 'DRB%',
    'AST per game', 'AST%',
    'STL per game', 'STL%',
    'BLK per game', 'BLK%',
    'TOV per game', 'TOV%',
    # 'OWS', 'DWS',
    # 'OBPM', 'DBPM',
    # 'VORP'
]

cluster_amount = 3

roster_df = pd.read_csv('merged_roster_2023121_0024_no_shooting.csv', delimiter=';')

filtered_df = filter_data(df=roster_df)

selected_df = select_features(df=filtered_df, selected_features=features)

overall_columns = [
    'MP per game', 'PTS per game', 'FG per game', '2P per game', '3P per game', 'FT per game',
    'ORB per game', 'DRB per game', 'AST per game', 'STL per game', 'BLK per game', 'TOV per game'
]
shooting_columns = ['FG%', '2P%', '3P%', 'FT%', 'eFG%', 'TS%']
advanced_columns = ['OWS', 'DWS', 'USG%', 'PER']

scale_data = False
normalize_data = True
save_file = True

clustered_kmeans_df = kmeans_clustering(
    filtered_df=filtered_df,
    selected_df=selected_df,
    scale_data=scale_data,
    normalize_data=normalize_data,
    n_clusters=cluster_amount,
    save_file=save_file
)

clustered_hierarchical_df = hierarchical_clustering(
    filtered_df=filtered_df,
    selected_df=selected_df,
    scale_data=scale_data,
    normalize_data=normalize_data,
    n_clusters=cluster_amount,
    save_file=save_file
)

clustered_gmm_df = gmm_clustering(
    filtered_df=filtered_df,
    selected_df=selected_df,
    scale_data=scale_data,
    normalize_data=normalize_data,
    n_clusters=cluster_amount,
    save_file=save_file
)

spectral_gmm_df = spectral_clustering(
    filtered_df=filtered_df,
    selected_df=selected_df,
    scale_data=scale_data,
    normalize_data=normalize_data,
    n_clusters=cluster_amount,
    save_file=save_file
)

# plot kmeans
plot_cluster_stats(
    clustered_kmeans_df,
    overall_columns,
    title_prefix='K-Means ',
    filename='plot_kmeans_1_overall'
)
plot_cluster_stats(
    clustered_kmeans_df,
    shooting_columns,
    title_prefix='K-Means ',
    filename='plot_kmeans_2_shooting'
)
plot_cluster_stats(
    clustered_kmeans_df,
    advanced_columns,
    title_prefix='K-Means ',
    filename='plot_kmeans_3_advanced'
)
plot_3d_cluster(
    clustered_kmeans_df,
    'AST per game',
    'TRB per game',
    'PTS per game',
    title_prefix='K-Means ',
    filename='plot_kmeans_4_3d'
)

# plot aglomerative
plot_cluster_stats(
    clustered_hierarchical_df,
    overall_columns,
    title_prefix='Hierarchical ',
    filename='plot_hierarchical_1_overall'
)
plot_cluster_stats(
    clustered_hierarchical_df,
    shooting_columns,
    title_prefix='Hierarchical ',
    filename='plot_hierarchical_2_shooting'
)
plot_cluster_stats(
    clustered_hierarchical_df,
    advanced_columns,
    title_prefix='Hierarchical ',
    filename='plot_hierarchical_3_advanced'
)
plot_3d_cluster(
    spectral_gmm_df,
    'AST per game',
    'TRB per game',
    'PTS per game',
    title_prefix='Hierarchical ',
    filename='plot_hierarchical_4_3d'
)

# gmm
plot_cluster_stats(
    clustered_gmm_df,
    overall_columns,
    title_prefix='GMM ',
    filename='plot_gmm_1_overall'
)
plot_cluster_stats(
    clustered_gmm_df,
    shooting_columns,
    title_prefix='GMM ',
    filename='plot_gmm_2_shooting'
)
plot_cluster_stats(
    clustered_gmm_df,
    advanced_columns,
    title_prefix='GMM ',
    filename='plot_gmm_3_advanced'
)
plot_3d_cluster(
    spectral_gmm_df,
    'AST per game',
    'TRB per game',
    'PTS per game',
    title_prefix='GMM ',
    filename='plot_gmm_4_3d'
)

# spectral
plot_cluster_stats(
    spectral_gmm_df,
    overall_columns,
    title_prefix='Spectral ',
    filename='plot_spectral_1_overall'
)
plot_cluster_stats(
    spectral_gmm_df,
    shooting_columns,
    title_prefix='Spectral ',
    filename='plot_spectral_2_shooting'
)
plot_cluster_stats(
    spectral_gmm_df,
    advanced_columns,
    title_prefix='Spectral ',
    filename='plot_spectral_3_advanced'
)
plot_3d_cluster(
    spectral_gmm_df,
    'AST per game',
    'TRB per game',
    'PTS per game',
    title_prefix='Spectral ',
    filename='plot_spectral_4_3d'
)
