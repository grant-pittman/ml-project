# importing app dependencies
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
from streamlit_folium import folium_static
import folium
from sklearn.mixture import GaussianMixture
import geopandas as gpd 
from math import pi

data_location = "cleaned_data.csv"


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page", ["Homepage", "Data", "Regression", "Clustering", "Player Stats"]
    )

    if page == "Homepage":
        # some formatting for the webapp
        st.title("FIFA 2018 Player Skills")
        img = "media/FIFA.jpg"
        st.image(img, width=400)
        st.markdown(
            """
        This is a project that uses ML to determine player pay based on their differnt skill levels
        """
        )
        st.info("These containers look nice")

    if page == "Data":

        # lets user pick how many rows they want to see in the app
        def load_data(nrows):
            data = pd.read_csv(data_location, encoding="latin1", nrows=nrows)
            return data

        # allows user to toggle visibility of the data
        if st.checkbox("Show raw data"):
            with st.beta_container():
                data_points = st.slider("data points", 0, 100, 50)
                data = load_data(data_points)
                st.subheader("Raw data")
                st.dataframe(data)

        map = folium.Map(location=['7.54', '-5.5471'], tiles="Stamen Toner", zoom_start=3)

        df = pd.read_csv("cleaned_data.csv")
        grouped_by_country = df.groupby("Nationality", as_index=False)["Name"].count()

        grouped_by_avg_value = df.groupby('Nationality', as_index=False)['Value'].mean()

        grouped_by_country.rename(columns={'Name': 'Player Count'}, inplace=True)

        country_lat_long = pd.read_csv('country_lat_long.csv', encoding='cp1252')


        merged_df = grouped_by_avg_value.merge(country_lat_long, left_on='Nationality', right_on='name', how='left')

        merged_df.loc[
            merged_df["Nationality"] == "Bosnia Herzegovina", "latitude"
        ] = 43.915886
        merged_df.loc[
            merged_df["Nationality"] == "Bosnia Herzegovina", "longitude"
        ] = 17.679076
        merged_df.loc[
            merged_df["Nationality"] == "Central African Rep.", "latitude"
        ] = 6.611111
        merged_df.loc[
            merged_df["Nationality"] == "Central African Rep.", "longitude"
        ] = 20.939444
        merged_df.loc[merged_df["Nationality"] == "DR Congo", "latitude"] = -4.038333
        merged_df.loc[merged_df["Nationality"] == "DR Congo", "longitude"] = 21.758664
        merged_df.loc[merged_df["Nationality"] == "England", "latitude"] = 55.378051
        merged_df.loc[merged_df["Nationality"] == "England", "longitude"] = -3.435973
        merged_df.loc[merged_df["Nationality"] == "Ivory Coast", "latitude"] = 7.54
        merged_df.loc[merged_df["Nationality"] == "Ivory Coast", "longitude"] = -5.5471
        merged_df.loc[
            merged_df["Nationality"] == "Korea Republic", "latitude"
        ] = 35.907757
        merged_df.loc[
            merged_df["Nationality"] == "Korea Republic", "longitude"
        ] = 127.766922
        merged_df.loc[merged_df["Nationality"] == "Scotland", "latitude"] = 56.4907
        merged_df.loc[merged_df["Nationality"] == "Scotland", "longitude"] = -4.2026
        merged_df.loc[merged_df["Nationality"] == "Wales", "latitude"] = 52.1307
        merged_df.loc[merged_df["Nationality"] == "Wales", "longitude"] = -3.7837

        for i in range(len(merged_df)):
            folium.Circle(
            location=[merged_df.iloc[i]['latitude'], merged_df.iloc[i]['longitude']],
            popup=f"Country: {merged_df.iloc[i]['Nationality']}, Avg. Value: {round(merged_df.iloc[i]['Value'],1)}",
            radius=float(merged_df.iloc[i]['Value']*5000),
            color='crimson',
            fill=True,
            fill_color='crimson',
            fill_opacity=0.8
            ).add_to(map)

        folium_static(map)

    if page == "Clustering":
        df = pd.read_csv(data_location)

        # only considering FIFA stats for clustering
        features_to_cluster = df.loc[:, "Crossing":"GKReflexes"].columns

        cluster_df = df.loc[:, features_to_cluster]
        cluster_df["Name"] = df["Name"]

        # generating a list of features to append to a user selection drop down menu
        feature_list = list(cluster_df.columns)
        feature_list.remove("Name")

        # sidebar multiselection to allow user to pick which features to use for clustering
        st.info("Please choose two features to compare. Consider using the most important features determined by regression")
        chosen_features = st.multiselect(
            "", feature_list
        )
        if len(chosen_features) < 2:
            st.stop()

        chosen_feature1 = chosen_features[0]
        chosen_feature2 = chosen_features[1]

        # df is filtered to two user inpits plus name column
        df_chosen = cluster_df[[chosen_feature1, chosen_feature2, "Name"]]

        with st.beta_expander("Kmeans"):


            # next user selects the k value for clustering
            st.info("Please choose a k value to change the chart")
            k = st.slider("", 2, 8, 5)

            st.info("Or if you dont know, use the auto value")
        
            if st.button("Auto value"):
                st.info("Okay well use the auto value now")
                #using a silhouette score to determine the best k-value 
                kmeans_per_k = [KMeans(n_clusters=k, random_state=0).fit(df_chosen.drop('Name', axis=1)) for k in range(2,10)]
                silhouette_scores = [silhouette_score(df_chosen.drop('Name', axis=1), model.labels_) for model in kmeans_per_k]
                #have not been able to get Tommy's original matplot code to work

                # plt.figure(figsize=(8, 5))
                # y = np.arange(2,10)
                # fig4 = plt.plot(y, silhouette_scores)
                # plt.xlabel("Number of Clusters (k)", fontsize=14)
                # plt.ylabel("Silhouette score", fontsize=14)
                # st.pyplot(fig4)

                #simplified chart that is correctly rendering
                y = np.arange(2,10)
                chart_data = pd.DataFrame(silhouette_scores,y)
                st.line_chart(chart_data)
                top_score = silhouette_scores.index(max(silhouette_scores))
                best_k_value = y[top_score]
                st.info('Here is the score for each K-value and you can see why we picked ' + str(best_k_value))

                kmeans = KMeans(n_clusters=best_k_value, random_state=0)
                y_pred = kmeans.fit_predict(df_chosen.drop("Name", axis=1))


            else:
                kmeans = KMeans(n_clusters=k, random_state=0)
                y_pred = kmeans.fit_predict(df_chosen.drop("Name", axis=1))

            # code adapted from Aurelien Geron's book 'Hands on Machine Learning with Scikit-Learn, Keras, and TensorFlow'
            # below is for Matplot

            def plot_data(df_chosen):
                plt.plot(df_chosen.loc[:,chosen_feature1], df_chosen.loc[:,chosen_feature2], 'k.', markersize=10)

            def plot_centroids(centroids, circle_color='seagreen', cross_color='k'):
                plt.scatter(centroids[:, 0], centroids[:, 1], s=30, linewidths=8,
                            color=circle_color, zorder=10, alpha=0.5)
                #plt.scatter(centroids[:, 0], centroids[:, 1], s=50, linewidths=50,
                #           color=cross_color, zorder=11, alpha=1)

            def plot_decision_boundaries(clusterer, df_chosen, resolution=1000, show_centroids=True,
                                        show_xlabels=True, show_ylabels=True, player=None):
                mins = df_chosen[[chosen_feature1, chosen_feature2]].min(axis=0) - 0.1
                maxs = df_chosen[[chosen_feature1, chosen_feature2]].max(axis=0) + 0.1
                xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                                    np.linspace(mins[1], maxs[1], resolution))
                Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                            cmap="viridis", alpha=0.7)
                plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                            linewidths=1, colors='k')
                
                
                plot_data(df_chosen)


                if player:
                    plt.scatter(df_chosen.loc[df_chosen['Name'] == player, chosen_feature1], df_chosen.loc[df_chosen['Name'] == player, chosen_feature2], color='red', s=700, marker='*', edgecolors='red', linewidths=3)

                if show_centroids:
                    plot_centroids(clusterer.cluster_centers_)

                if show_xlabels:
                    plt.xlabel(f"{chosen_feature1}", fontsize=14)
                else:
                    plt.tick_params(labelbottom=False)
                if show_ylabels:
                    plt.ylabel(f"{chosen_feature2}", fontsize=14, rotation=90)
                else:
                    plt.tick_params(labelleft=False)

            plt.figure(figsize=(8, 4))
            player = st.text_input("Which player do you want to Highlight?", 'Cristiano Ronaldo')

            plot = plot_decision_boundaries(kmeans, df_chosen, player = player)

            # needed to remove PyplotGlobalUseWarning
            st.set_option("deprecation.showPyplotGlobalUse", False)

            st.pyplot(plot)



            # below is for Plot.ly but it is not working yet

            # with st.beta_expander("Plotly"):

            #     def plot_decision_boundaries_plotly(clusterer, df_chosen, player=None):
            #         x_min, x_max = (
            #             df_chosen.loc[:, chosen_feature1].min() - 1,
            #             df_chosen.loc[:, chosen_feature1].max() + 1,
            #         )
            #         y_min, y_max = (
            #             df_chosen.loc[:, chosen_feature2].min() - 1,
            #             df_chosen.loc[:, chosen_feature2].max() + 1,
            #         )
            #         # maxs = df_chosen.max(axis=0) + 0.1
            #         xx, yy = np.meshgrid(
            #             np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
            #         )
            #         y_ = np.arange(y_min, y_max, 0.02)
            #         Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
            #         Z = Z.reshape(xx.shape)

            #         trace1 = go.Heatmap(
            #             x=xx[0], y=y_, z=Z, colorscale="Viridis", showscale=True
            #         )

            #         trace2 = go.Scatter(
            #             x=df_chosen.loc[:, chosen_feature1],
            #             y=df_chosen.loc[:, chosen_feature2],
            #             mode="markers",
            #             text=df["Name"],
            #             marker=dict(
            #                 size=10,
            #                 color=df[chosen_feature1],
            #                 colorscale="Viridis",
            #                 line=dict(color="black", width=1),
            #             ),
            #         )

            #         if player:
            #             trace3 = go.Scatter(
            #                 x=df_chosen.loc[df_chosen["Name"] == player, chosen_feature1],
            #                 y=df_chosen.loc[df_chosen["Name"] == player, chosen_feature2],
            #                 mode="markers",
            #                 marker=dict(
            #                     size=20, color="red", line=dict(color="black", width=2)
            #                 ),
            #             )
            #             data = [trace1, trace2, trace3]
            #         else:
            #             data = [trace1, trace2]

            #         layout = go.Layout(
            #             autosize=True,
            #             title="K-Means",
            #             hovermode="closest",
            #             showlegend=False,
            #         )

            #         # data = [trace1, trace2]
            #         # fig = go.Figure(data=data, layout=layout)

            #     player = st.text_input("which player are you interested in?", "L. Massi")

            #     plot2 = plot_decision_boundaries_plotly(kmeans, df_chosen, player=player)

            #     st.plotly_chart(plot2)
        with st.beta_expander("Gausian Blur"):
            gm = GaussianMixture(n_components=4, n_init=10, random_state=0)
            gm.fit(df_chosen.drop('Name', axis=1))


            from matplotlib.colors import LogNorm

            def plot_gaussian_mixture(clusterer, df_chosen, resolution=1000, show_ylabels=True, player=None, plot_anomalies = False, density_cutoff = 2):
                mins = df_chosen.drop('Name', axis=1).min(axis=0) - 0.1
                maxs = df_chosen.drop('Name', axis=1).max(axis=0) + 0.1
                xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                                    np.linspace(mins[1], maxs[1], resolution))
                Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                plt.contourf(xx, yy, Z,
                            norm=LogNorm(vmin=1.0, vmax=30.0),
                            levels=np.logspace(0, 2, 12), cmap='viridis')
                plt.contour(xx, yy, Z,
                            norm=LogNorm(vmin=1.0, vmax=30.0),
                            levels=np.logspace(0, 2, 12),
                            linewidths=1, colors='k')

                Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contour(xx, yy, Z,
                            linewidths=2, colors='r', linestyles='dashed')
                
                plt.plot(df_chosen.loc[:, chosen_feature1], df_chosen.loc[:, chosen_feature2], 'k.', markersize=10)
                #plot_centroids(clusterer.means_, clusterer.weights_)
                if player:
                    plt.scatter(df_chosen.loc[df_chosen['Name'] == player, chosen_feature1], df_chosen.loc[df_chosen['Name'] == player, chosen_feature2], color='red', s=700, marker='*', edgecolors='red', linewidths=3)
                if plot_anomalies:
                    densities = clusterer.score_samples(df_chosen.drop('Name', axis=1))
                    density_threshold = np.percentile(densities, 2)
                    anomalies = df_chosen[densities < density_threshold]
                    plt.scatter(anomalies.loc[:, chosen_feature1], anomalies.loc[:, chosen_feature2], color='gray', s=400, edgecolors='gray', linewidths=3)

                plt.xlabel("$x_1$", fontsize=14)
                if show_ylabels:
                    plt.ylabel("$x_2$", fontsize=14, rotation=0)
                else:
                    plt.tick_params(labelleft=False)
            
            fig5 = plot_gaussian_mixture(gm, df_chosen, player='L. Messi', plot_anomalies=True)

            st.pyplot(fig5)

    if page == "Regression":
        data = pd.read_csv(data_location)

        target = data["Value"]
        features = data.drop(
            [
                "ID",
                "Value",
                "Name",
                "Nationality",
                "Club",
                "Wage",
                "Preferred Foot",
                "Work Rate",
                "Position",
                "Jersey Number",
                "Joined",
                "Contract Valid Until",
                "Feet"
            ],
            axis=1,
        )

        # Set column headers as feature names
        feature_names = features.columns

        
        n_estimators = st.slider("Choose the number of trees", 0, 1000, 500)

    
        # RandomForestRegressor. Best for analyzing and ranking several features.
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf = rf.fit(features, target)

        importances = rf.feature_importances_

        df = pd.DataFrame({'Features' : feature_names, 'Scores' : importances})


        def create_bar_chart():
            top_features = df.nlargest(10, "Scores")
            a = top_features["Scores"]
            b = top_features["Features"]
            plt.bar(b,a)
            
            plt.xlabel("Features")
            plt.xticks(b, rotation='vertical')
            plt.ylabel("Score")

        st.info("Here are the most important features for the model")
       

        col1, col2 = st.beta_columns(2)

        with col1:
            st.dataframe(df.sort_values("Scores", ascending = False))

        with col2:
            plot7 = create_bar_chart()

            st.pyplot(plot7)

        selected_features = st.multiselect("Which features do you want to use in the model?", feature_names)

        y = target.values.reshape(-1, 1)

        X = features[
            selected_features
        ]
        

        if st.button("Train model"):
            # Split training and testing data.
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

            # Normalize data.
            from sklearn.preprocessing import StandardScaler

            X_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train)

            X_train_scaled = X_scaler.transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)
            y_train_scaled = y_scaler.transform(y_train)
            y_test_scaled = y_scaler.transform(y_test)

            # Create linear regression model.
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X_train_scaled, y_train_scaled)

            # Plot results
            predictions = model.predict(X_test_scaled)
            model.fit(X_train_scaled, y_train_scaled)
            plot3 = plt.scatter(
                model.predict(X_train_scaled),
                model.predict(X_train_scaled) - y_train_scaled,
                c="green",
                label="Training Data",
            )
            plt.scatter(
                model.predict(X_test_scaled),
                model.predict(X_test_scaled) - y_test_scaled,
                c="orange",
                label="Testing Data",
            )
            plt.legend()
            plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
            plt.title("Residual Plot")
            plt.show()

            st.pyplot()

            model.fit(X_train, y_train.ravel())

            st.markdown(f"Training Data Score: {model.score(X_train, y_train)}")
            st.markdown(f"Testing Data Score: {model.score(X_test, y_test)}")


            top_players = data.nlargest(10, "Value")
            top_name = top_players["Name"]
            top_value = top_players["Value"]

            top_features = top_players[selected_features]

            predictions = np.round(model.predict(top_features),1)

           
            def value_bar_chart():
                N = 10
                men_means = predictions

                ind = np.arange(N)  # the x locations for the groups
                width = 0.35       # the width of the bars

                fig, ax = plt.subplots()
                rects1 = ax.bar(ind, men_means, width, color='r')

                women_means = top_value
                rects2 = ax.bar(ind + width, women_means, width, color='y')

                # add some text for labels, title and axes ticks
                ax.set_ylabel('Value')
                ax.set_title('Predicted V.S. Actual Player Value')
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(top_name)
                plt.xticks(rotation=45)

                ax.legend((rects1[0], rects2[0]), ('Prediction', 'Actual'))


                def autolabel(rects):
                    """
                    Attach a text label above each bar displaying its height
                    """
                    for rect in rects:
                        height = rect.get_height()
                        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                                '%d' % int(height),
                                ha='center', va='bottom')

                autolabel(rects1)
                autolabel(rects2)

                plt.show()
            plot8 = value_bar_chart()
            st.pyplot(plot8)
    if page == "Player Stats":
        df = pd.read_csv('cleaned_data.csv')
        most_important_features = ['Age', 'Potential', 'Finishing', 'Reactions', 'Dribbling', 'BallControl', 'LongShots', 'Volleys', 'Vision']
        df_subset = df.loc[:,['Name']+most_important_features]
        categories = list(df_subset)[2:]
        N = len(categories)

        chosen_name = st.text_input("Choose a player")

        if chosen_name:
            values = df_subset.loc[df_subset['Name'] == chosen_name,:].drop(['Name', 'Age'], axis=1).values.flatten().tolist()
            values += values[:1]
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            def create_radar_chart():
                plt.figure(figsize=(12,8))

                ax = plt.subplot(111, polar=True)

                plt.xticks(angles[:-1], categories, color='grey', size=8)

                ax.set_rlabel_position(0)

                plt.yticks([40, 60, 80], ["40", "60", "80"], color='grey', size=7)
                plt.ylim(0,100)

                ax.plot(angles, values, linewidth=1, linestyle='solid')

                ax.fill(angles, values, 'b', alpha=0.1);
            
            plot9 = create_radar_chart()
            st.pyplot(plot9)

            specific_player = df.loc[df["Name"] == chosen_name]
            contract_value = specific_player["Value"].item()

            st.info(f"{chosen_name} 's contract value is {contract_value}") 
        else:
            st.stop()





if __name__ == "__main__":
    main()

st.set_option('deprecation.showPyplotGlobalUse', False)


