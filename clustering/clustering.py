from datetime import datetime

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize, StandardScaler
from yellowbrick.cluster import KElbowVisualizer


def filter_data(
        df: pd.DataFrame,
        min_games=21,
        min_mp_per_game=12.0,
        min_pts_per_game=0.0,
        min_ast_per_game=0.0,
        min_trb_per_game=0.0
):
    # Fill missing values with 0.0
    df.fillna(0.0, inplace=True)

    df_filtered = df[df['G'] >= min_games]
    df_filtered = df_filtered[df_filtered['MP per game'] >= min_mp_per_game]
    df_filtered = df_filtered[df_filtered['PTS per game'] >= min_pts_per_game]
    df_filtered = df_filtered[df_filtered['AST per game'] >= min_ast_per_game]
    df_filtered = df_filtered[df_filtered['TRB per game'] >= min_trb_per_game]

    return df_filtered


def select_features(df: pd.DataFrame, selected_features: list):
    return df[selected_features]


def kmeans_clustering(
        filtered_df: pd.DataFrame,
        selected_df: pd.DataFrame,
        scale_data=False,
        scaler=StandardScaler(),
        normalize_data=False,
        n_clusters=-1,
        save_file=False,
        method_prefix='kmeans',
):
    filtered_df_copy = filtered_df.copy()

    # Standardize the data
    if scale_data:
        df_scaled = scaler.fit_transform(selected_df)
    elif normalize_data:
        df_scaled = normalize(selected_df)
    else:
        df_scaled = selected_df

    if n_clusters <= 0:
        # silhouette_scores = []
        # cluster_range = range(2, 11)
        #
        # for n_clusters in cluster_range:
        #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, random_state=42)
        #     clusters = kmeans.fit_predict(df_scaled)
        #     silhouette_avg = silhouette_score(df_scaled, clusters)
        #     silhouette_scores.append(silhouette_avg)
        #
        # # Plot silhouette scores for different cluster numbers
        # plt.plot(cluster_range, silhouette_scores, marker='o')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Silhouette Score')
        # plt.title('Silhouette Analysis for Optimal Number of Clusters')
        # plt.show()
        #
        # cluster_range = range(2, 10)
        #
        # fig, ax = plt.subplots(4, 2, figsize=(12, 12))  # Adjust the figsize as needed
        #
        # for i, n_clusters in enumerate(cluster_range):
        #     model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, random_state=42)
        #     q, mod = divmod(i, 2)
        #
        #     visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=ax[i // 2, i % 2])
        #     visualizer.fit(df_scaled)
        #
        #     ax[i // 2, i % 2].set_title(f'{n_clusters} Clusters')
        #
        # plt.suptitle('Silhouette Plots', fontsize=16)
        # plt.tight_layout()
        # plt.show()
        #
        # n_clusters = int(input("Enter the optimal number of clusters (k): "))

        # # Elbow Method to find optimal k
        # sse = []
        # k_values = range(1, 13)
        #
        # for k in k_values:
        #     kmeans = KMeans(n_clusters=k, random_state=42)
        #     kmeans.fit(df_scaled)
        #     sse.append(kmeans.inertia_)
        #
        # # Calculate SSE differences
        # sse_difference = [sse[i] - sse[i + 1] for i in range(len(sse) - 1)]
        # sse_difference_2unit = [sse[i] - 2 * sse[i + 1] + sse[i + 2] for i in range(len(sse) - 2)]
        #
        # # Plot SSE, SSE Differences, and SSE 2-Unit Differences
        # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
        #
        # # Plot SSE
        # axes[0].plot(k_values, sse, marker='o')
        # axes[0].set_title('Elbow Method for Optimal k - SSE')
        # axes[0].set_xlabel('Number of Clusters (k)')
        # axes[0].set_ylabel('Sum of Squared Errors (SSE)')
        #
        # # Plot SSE Differences
        # axes[1].plot(k_values[:-1], sse_difference, marker='o')
        # axes[1].set_title('SSE Rolling Difference')
        # axes[1].set_xlabel('Number of Clusters (k)')
        # axes[1].set_ylabel('SSE Difference')
        #
        # # Plot SSE 2-Unit Differences
        # axes[2].plot(k_values[:-2], sse_difference_2unit, marker='o')
        # axes[2].set_title('SSE Rolling 2-Unit Difference')
        # axes[2].set_xlabel('Number of Clusters (k)')
        # axes[2].set_ylabel('SSE 2-Unit Difference')
        #
        # plt.tight_layout()
        # plt.show()

        # # Based on the elbow plot, choose the optimal number of clusters (k)
        # n_clusters = int(input("Enter the optimal number of clusters (k): "))

        # Instantiate the KMeans model
        model = KMeans()

        # Instantiate the visualizer with the KMeans model
        visualizer = KElbowVisualizer(model, k=(1, 11), timings=False)

        # Fit the visualizer to the data
        visualizer.fit(df_scaled)

        # Visualize the elbow plot
        visualizer.show()

        # Based on the elbow plot, the optimal number of clusters is automatically determined
        n_clusters = visualizer.elbow_value_

        print(f"Optimal number of clusters (k): {n_clusters}")

    # Apply k-means clustering with the chosen k
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    # Add the cluster labels to the original DataFrame
    filtered_df_copy.loc[:, 'Cluster'] = clusters

    if save_file:
        save_to_file(filtered_df_copy, method_prefix)

    plot_pca(clusters, df_scaled, method_prefix)

    print_representatives(filtered_df_copy, n_clusters)

    return filtered_df_copy


def hierarchical_clustering(
        filtered_df: pd.DataFrame,
        selected_df: pd.DataFrame,
        scale_data=False,
        scaler=StandardScaler(),
        normalize_data=False,
        n_clusters: int = 2,
        save_file: bool = False,
        method_prefix='hierarchical',
):
    filtered_df_copy = filtered_df.copy()

    # Standardize the data
    if scale_data:
        df_scaled = scaler.fit_transform(selected_df)
    elif normalize_data:
        df_scaled = normalize(selected_df)
    else:
        df_scaled = selected_df

    # Apply Agglomerative Clustering with the specified number of clusters
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = agg_clustering.fit_predict(df_scaled)

    # Add the cluster labels to the original DataFrame
    filtered_df_copy.loc[:, 'Cluster'] = clusters

    if save_file:
        save_to_file(filtered_df_copy, method_prefix)

    plot_pca(clusters, df_scaled, method_prefix)

    print_representatives(filtered_df_copy, n_clusters)

    return filtered_df_copy


def gmm_clustering(
        filtered_df: pd.DataFrame,
        selected_df: pd.DataFrame,
        scale_data=False,
        scaler=StandardScaler(),
        normalize_data=False,
        n_clusters: int = 2,
        save_file: bool = False,
        method_prefix='gmm',
):
    filtered_df_copy = filtered_df.copy()

    # Standardize the data
    if scale_data:
        df_scaled = scaler.fit_transform(selected_df)
    elif normalize_data:
        df_scaled = normalize(selected_df)
    else:
        df_scaled = selected_df

    # Apply Gaussian Mixture Model with the specified number of clusters
    gmm = GaussianMixture(n_components=n_clusters)
    clusters = gmm.fit_predict(df_scaled)

    # Add the cluster labels to the original DataFrame
    filtered_df_copy.loc[:, 'Cluster'] = clusters

    if save_file:
        save_to_file(filtered_df_copy, method_prefix)

    plot_pca(clusters, df_scaled, method_prefix)

    print_representatives(filtered_df_copy, n_clusters)

    return filtered_df_copy


def spectral_clustering(
        filtered_df: pd.DataFrame,
        selected_df: pd.DataFrame,
        scale_data=False,
        scaler=StandardScaler(),
        normalize_data=False,
        n_clusters: int = 2,
        save_file: bool = False,
        method_prefix='spectral',
):
    filtered_df_copy = filtered_df.copy()

    # Standardize the data
    # df_scaled = scaler.fit_transform(selected_df)
    if scale_data:
        df_scaled = scaler.fit_transform(selected_df)
    elif normalize_data:
        df_scaled = normalize(selected_df)
    else:
        df_scaled = selected_df

    # Apply Spectral Clustering with the specified number of clusters
    gmm = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, eigen_solver='arpack')
    clusters = gmm.fit_predict(df_scaled)

    # Add the cluster labels to the original DataFrame
    filtered_df_copy.loc[:, 'Cluster'] = clusters

    if save_file:
        save_to_file(filtered_df_copy, method_prefix)

    plot_pca(clusters, df_scaled, method_prefix)

    print_representatives(filtered_df_copy, n_clusters)

    return filtered_df_copy


def save_to_file(filtered_df_copy, method_prefix):
    # Save the clustered data to a CSV file
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = f'{method_prefix}_{current_time}.csv'
    filtered_df_copy.to_csv(csv_filename, index=False)
    print(f"Merged roster saved to {csv_filename}")


def plot_pca(clusters, df_scaled, method_prefix=None):
    # Use PCA for dimensionality reduction (2D plot)
    plt.clf()

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=['PCA1', 'PCA2'])
    # Combine PCA results with cluster labels
    df_pca.loc[:, 'Cluster'] = clusters
    # Plot the clusters in 2D
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='viridis', s=50)
    plt.title('Clusters in 2D PCA Space')

    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    if method_prefix:
        plt.savefig(f'plot_{method_prefix}_0_pca_{current_time}.webp')
    else:
        plt.show()


def print_representatives(filtered_df_copy, n_clusters):
    for cluster_label in range(n_clusters):
        cluster_data = filtered_df_copy[filtered_df_copy['Cluster'] == cluster_label]
        # ax.scatter(cluster_data['PTS per game'], cluster_data['AST per game'], cluster_data['TRB per game'],
        #            label=f'Cluster {cluster_label}')

        # Print 3 best representative records for each cluster
        print(f"\nCluster {cluster_label} - 3 Best Representative Records\nby PTS per game:")
        cluster_representatives_pts_per_game = cluster_data.nlargest(3, 'PTS per game')
        print(cluster_representatives_pts_per_game[
                  ['Player', 'Team', 'Season', 'G', 'PTS per game', '2P%', '3P%',
                   'ORB per game', 'DRB per game', 'TRB per game', 'AST per game', 'PER']
              ].to_markdown())
        print(f"\nby TRB:")
        cluster_representatives_trb_per_game = cluster_data.nlargest(3, 'TRB per game')
        print(cluster_representatives_trb_per_game[
                  ['Player', 'Team', 'Season', 'G', 'PTS per game', '2P%', '3P%',
                   'ORB per game', 'DRB per game', 'TRB per game', 'AST per game', 'PER']
              ].to_markdown())
        print(f"\nby AST:")
        cluster_representatives_ast_per_game = cluster_data.nlargest(3, 'AST per game')
        print(cluster_representatives_ast_per_game[
                  ['Player', 'Team', 'Season', 'G', 'PTS per game', '2P%', '3P%',
                   'ORB per game', 'DRB per game', 'TRB per game', 'AST per game', 'PER']
              ].to_markdown())
        print(f"\nby PER:")
        cluster_representatives_per = cluster_data.nlargest(3, 'PER')
        print(cluster_representatives_per[
                  ['Player', 'Team', 'Season', 'G', 'PTS per game', '2P%', '3P%',
                   'ORB per game', 'DRB per game', 'TRB per game', 'AST per game', 'PER']
              ].to_markdown())


def plot_cluster_stats(df, columns_of_interest, cluster_column='Cluster', title_prefix='', filename=None, sns=None):
    num_columns = len(columns_of_interest)
    num_clusters = df[cluster_column].nunique()

    ncols = int(num_columns / 2 + num_columns % 2)

    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(24, 8))

    axes = axes.flatten()

    for idx, column in enumerate(columns_of_interest):
        ax = axes[idx]

        # Plot cluster minimum and maximum using boxplot
        sns.boxplot(x=cluster_column, y=column, data=df, showfliers=False, ax=ax)

        ax.set_title(f'{title_prefix} for {column}')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(column)

    plt.tight_layout()

    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    if filename:
        plt.savefig(f'{filename}_{current_time}.webp')
    else:
        plt.show()


def plot_3d_cluster(df, x, y, z, cluster_column='Cluster', title_prefix='', filename=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    clusters = df[cluster_column].unique()

    for cluster_label in clusters:
        cluster_data = df[df[cluster_column] == cluster_label]
        ax.scatter(cluster_data[x], cluster_data[y], cluster_data[z], label=f'Cluster {cluster_label}')

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f'{title_prefix} 3D Scatter Plot')

    plt.legend()

    current_time = datetime.now().strftime("%Y%m%d_%H%M")

    if filename:
        plt.savefig(f'{filename}_{current_time}.webp')
    else:
        plt.show()
