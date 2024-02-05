#%% Cluster analysis (from analyze_calibration.py)
def run_cluster_analysis(df, do_save=True, plot_densities=False, cutoff=None):
    if cutoff is None:
        df_good = df
    else:
        df_good = df[df['mismatch'] < cutoff]

    variables_to_plot = df.columns.difference(['mismatch', 'index'])
    variables_to_plot = ['hpv_control_prob','sev_dist_par1']
    cmap = sns.color_palette("viridis_r", n_colors=3)

    sns.set(font_scale=1.15)
    if plot_densities:
        fig, axes = pl.subplots(8, 4, figsize=(20, 20))

        for i, ax in enumerate(axes.flat):
            if i < len(variables_to_plot):
                variable = variables_to_plot[i]
                # plot for each data df
                sns.kdeplot(data=df[df['mismatch'] < 2][variable], label='First samples (mismatch < '+str(2)+')', fill=True, common_norm=False, ax=ax, color=cmap[0], alpha=0.5)
                sns.kdeplot(data=df[df['mismatch'] < 1][variable], label='Second samples (mismatch < '+str(1)+')', fill=True, common_norm=False, ax=ax, color=cmap[1], alpha=0.5)
                sns.kdeplot(data=df[df['mismatch'] < 0.5][variable], label='Third samples (mismatch <'+str(0.5)+')', fill=True, common_norm=False, ax=ax, color=cmap[2], alpha=0.5)
            else:
                ax.set_axis_off()

        axes[0,0].legend()
        fig.tight_layout()
        # filename = f'{location}_calib{calib_filestem}_densities'
        # fig.savefig(f'{ut.figfolder}/{filename}.png')

    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    np.random.seed(0)

    data_df = df_good

    X = data_df.iloc[:, 2:]
    # X = data_df[variables_to_plot]
    Y = data_df['mismatch']

    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Getting silhouette_score
    from sklearn.metrics import silhouette_score

    silhouette_scores = []

    # Try different numbers of clusters
    for n in range(2, 10):
        gmm = GaussianMixture(n_components=n)
        gmm_fit = gmm.fit(X_scaled)
        cluster_label = gmm.predict(X_scaled)
        silhouette_scores.append(silhouette_score(X, cluster_label))

    print(silhouette_scores)
    # Plot the silhouette scores
    pl.figure(figsize=(10, 6))
    pl.plot(range(2, 10), silhouette_scores, marker='o')
    pl.xlabel('Number of Clusters')
    pl.ylabel('Silhouette Score')
    pl.title('Silhouette Analysis')
    pl.show()

    # Determine number of clusters from this
    # Create and fit the Gaussian Mixture Model
    n_components = np.arange(2, 10)[np.where(silhouette_scores == np.max(silhouette_scores))[0]][0]  # Number of clusters
    # n_components = 5
    gmm = GaussianMixture(n_components=n_components)
    gmm_fit = gmm.fit(X_scaled)

    # Obtain the predicted cluster labels
    cluster_labels = gmm.predict(X_scaled)

    data_df['cluster'] = cluster_labels

    # pl.hist(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    selected_points = {cluster: [] for cluster in unique_clusters}
    n_samples = 5

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        selected_index = cluster_indices[:n_samples]
        selected_point = X.iloc[selected_index, :].values
        selected_points[cluster] = selected_point

    cmap2 = sns.color_palette("coolwarm_r", n_colors=n_components)
    data_df = data_df.reset_index(drop=True)

    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    for i, variable in enumerate(data_df.columns.difference(['mismatch', 'index','cluster'])):
    # for i, variable in enumerate(variables_to_plot):
        ax = axes[i // 6, i % 6]
        #plot kde for each cluster
        for cluster in unique_clusters:
            sns.kdeplot(data=data_df[variable][data_df['cluster'] == cluster], label='Cluster '+str(cluster), fill=True, common_norm=False, ax=ax, color=cmap2[cluster], alpha=1)
        for cluster, points in selected_points.items():
            for point in points:
                if variable in df.columns:
                    ax.scatter(x=point[i], y=0.8*ax.get_ylim()[1], color=cmap2[cluster], alpha=1)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 1.03), ncol=3)
    plt.tight_layout()
    plt.show()

    # data_df.to_csv(f'{ut.resfolder}/{location}_data_df.csv', index=True)

    # See how each data point's cluster results.
    # pl.scatter(data_df.index, data_df.cluster)
    # pl.title('Clustering assignment')
    # pl.show()
    return data_df

cluster_df = run_cluster_analysis(calib100_df, cutoff=1.8)