import streamlit as st
import pandas as pd
import geopandas as gpd
import pyproj
import osmnx as ox
from shapely.geometry import Point
from shapely.ops import transform

import folium
from streamlit_folium import st_folium

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title=" New Fast Food Location Predictor ", layout="wide", initial_sidebar_state="auto")


# --------------------- HEADER (TWO COLUMNS) --------------------------
header_col1, header_col2 = st.columns([1, 1])
with header_col1:
    st.title("🍔 New Franchise Fast Food Location Predictor – Sydney")
    st.markdown("<span style='font-size:18px; color:white;'>This project uses geospatial analysis and machine learning to identify optimal locations for new fast food franchise outlets in Greater Sydney. By combining open geospatial data, census statistics, and predictive modeling, the workflow helps franchise owners and analysts make data-driven decisions about expansion.</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **Input Data:**
    - 2021 Australian Census data for Greater Sydney (population, suburb boundaries)
    - OpenStreetMap data for fast food locations (McDonald's, KFC, Subway)

    **Key Variables Used:**
    - `brand`: Fast food franchise to analyze (selectable in sidebar)
    - `POP_DENSITY_THRESHOLD`: Minimum population density for candidate suburbs (default: 1000)
    - `DISTANCE_THRESHOLD_KM`: Minimum distance from existing outlets for new recommendations (default: 3.0 km)
    - `density_weight` and `distance_weight`: User-adjustable weights for location scoring

    **Output:**
    - Top 3 recommended suburbs for new outlet (based on population density, distance, and scoring)
    - Interactive map and table of results
    """)
    # Links to GitHub, Colab, and Kaggle horizontally
    link_col1, link_col2, link_col3 = st.columns([1,1,1])
    with link_col1:
        st.markdown('[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12xvCIPyhirh0Q1bNXtO_e6jN5wZmzaBi?usp=sharing)', unsafe_allow_html=True)
    with link_col2:
        st.markdown('[![GitHub Repo](https://img.shields.io/badge/GitHub-fast_food-blue?logo=github)](https://github.com/basanta11/fast_food)', unsafe_allow_html=True)
    with link_col3:
        st.markdown('[![View on Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/basantadahal/franchise-new-location-predictor)', unsafe_allow_html=True)

with header_col2:
    st.subheader("🏪 Select a Fast Food Brand")
    # Data loading for brand list (minimal, just for selectbox)
    with st.spinner("🔄 Fetching fast food data..."):
        ox.settings.log_console = False
        tags = {"amenity": "fast_food", "name": ["McDonald's", "KFC", "Subway"]}
        gdf = ox.features_from_place("Sydney, Australia", tags=tags)
        df = gdf[['name', 'geometry', 'brand', 'branch', 'addr:street']]
        df = df[df['brand'].notnull()]
        brand_list = df['brand'].value_counts().index.tolist()
    brand = st.selectbox("Select Brand", brand_list)
    st.subheader("🎯 Adjust Weights for Location Score")
    density_weight = st.slider("🏙️ Population Density Weight", 0, 10, 7)
    distance_weight = 10 - density_weight
    st.metric("📍 Distance Weight", value=distance_weight)



    test_filtered = df[df['brand'] == brand]
    if test_filtered.crs.is_geographic:
        test_filtered = test_filtered.to_crs(epsg=3857)

    test_filtered["geometry_centroid"] = test_filtered.geometry.centroid
    test_filtered["geometry_centroid_latlon"] = test_filtered["geometry_centroid"].to_crs(epsg=4326)

    # ------------------ LOAD SHAPEFILES --------------------

    sa2_gdf = gpd.read_file("data/SA2_2021_AUST_SHP_GDA2020/SA2_2021_AUST_GDA2020.shp")
    sa2_nsw = sa2_gdf[sa2_gdf['STE_NAME21'] == 'New South Wales']
    sa2 = sa2_nsw[sa2_nsw['GCC_NAME21'] == 'Greater Sydney'][['SA2_CODE21', 'SA2_NAME21', 'geometry']]

    census_data = pd.read_csv('data/2021Census_G01_NSW_SA2.csv')
    census = census_data[['SA2_CODE_2021', 'Tot_P_P']].copy()
    census['SA2_CODE21'] = census['SA2_CODE_2021'].astype(str)
    sa2 = sa2.merge(census, on='SA2_CODE21', how='left').to_crs(epsg=3577)

    sa2['area_km2'] = sa2['geometry'].area / 1e6
    sa2['pop_density'] = sa2['Tot_P_P'] / sa2['area_km2']

    # Match fast food to nearest suburb
    if test_filtered.crs != sa2.crs:
        sa2 = sa2.to_crs(test_filtered.crs)

    def get_nearest_suburb(point, gdf):
        distances = gdf.geometry.distance(point)
        return gdf.loc[distances.idxmin()]

    test_filtered['nearest_suburb'] = test_filtered['geometry_centroid'].apply(lambda pt: get_nearest_suburb(pt, sa2)['SA2_NAME21'])

    # Count and join fast food per suburb
    suburb_list = sa2.copy()
    fast_food = test_filtered[['name', 'geometry', 'nearest_suburb', 'geometry_centroid']]
    suburb_counts = fast_food["nearest_suburb"].value_counts().reset_index()
    suburb_counts.columns = ["suburb", "fast_food_count"]
    suburb_list = suburb_list.merge(suburb_counts, left_on='SA2_NAME21', right_on='suburb', how="left")
    suburb_list["fast_food_count"] = suburb_list["fast_food_count"].fillna(0).astype(int)

    # Distance from suburb to nearest fast food
    suburb_list_proj = suburb_list.to_crs(epsg=3857).copy()
    fast_food = fast_food.to_crs(suburb_list_proj.crs)
    suburb_list_proj['centroid'] = suburb_list_proj.geometry.centroid
    suburb_list_proj['min_dist_to_fastfood_km'] = suburb_list_proj['centroid'].apply(
        lambda x: fast_food.distance(x).min() / 1000
    )
    suburb_list_proj['area_km2'] = suburb_list_proj['geometry'].area / 1_000_000
    suburb_list_proj['pop_density'] = suburb_list_proj['Tot_P_P'] / suburb_list_proj['area_km2']

    # Normalize and calculate score
    scaler = MinMaxScaler()
    suburb_list_proj[['norm_density', 'norm_dist']] = scaler.fit_transform(
        suburb_list_proj[['pop_density', 'min_dist_to_fastfood_km']]
    )

    suburb_list_proj['location_score'] = (
        suburb_list_proj['norm_density'] * (density_weight / 10) +
        suburb_list_proj['norm_dist'] * (distance_weight / 10)
    )

    # Filter top areas
    POP_DENSITY_THRESHOLD = 1000
    DISTANCE_THRESHOLD_KM = 3.0

    candidate_suburbs = suburb_list_proj[
        (suburb_list_proj['pop_density'] > POP_DENSITY_THRESHOLD) &
        (suburb_list_proj['min_dist_to_fastfood_km'] > DISTANCE_THRESHOLD_KM)
    ].copy()
    top_areas = candidate_suburbs.sort_values(by='pop_density', ascending=False).head(3)
    top_areas['Nearest_franchise_location(in km)']=top_areas['min_dist_to_fastfood_km']
    top_areas['Suburb Name']=top_areas['SA2_NAME21']
    top_areas['Population density']=top_areas['pop_density']



    # ----------------- SHOW MAP + RECOMMENDATIONS (BELOW CONTROLS) -----------------------
    st.markdown("## 📍 Suggested Locations for New Outlet")
    st.dataframe(top_areas[['Suburb Name', 'Population density', 'Nearest_franchise_location(in km)', 'location_score']].reset_index(drop=True))
    with st.spinner("🔄 Rendering location map..."):
        fast_food_wm = fast_food.to_crs(epsg=3857)
        suburb_list_wm = suburb_list.to_crs(epsg=3857)
        top_areas_wm = top_areas.to_crs(epsg=3857)

        combined = pd.concat([fast_food_wm[['geometry']], top_areas_wm[['geometry']]], ignore_index=True)
        minx, miny, maxx, maxy = combined.total_bounds
        fig, ax = plt.subplots(figsize=(6, 5))  # Smaller image
        fast_food_wm.plot(ax=ax, color='blue', markersize=5)
        top_areas_wm.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)

        texts = []
        for idx, row in top_areas_wm.iterrows():
            texts.append(
                plt.text(
                    row.geometry.centroid.x + 5000 ,
                    row.geometry.centroid.y + 5000,
                    row['SA2_NAME21'],
                    fontsize=10, color='purple', ha='left', va='center',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
                )
            )

        adjust_text(
        texts,
        ax=ax,
        expand_text=(1.2, 1.5),
        expand_points=(2, 2),
        force_points=0.2,
        force_text=0.2,
        lim=500
    )
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
        ax.set_xlim(minx - 5000, maxx + 5000)
        ax.set_ylim(miny - 5000, maxy + 5000)
        ax.legend(handles=[
            Patch(facecolor='white', edgecolor='gray', label='Suburbs'),
            Line2D([0], [0], marker='o', color='w', label='Fast Food Venue', markerfacecolor='blue', markersize=6),
            Line2D([0], [0], marker='s', color='r', label='Top 3 Suggested Areas', markerfacecolor='none', markersize=10)
        ])
        ax.set_title("Top 3 Suggested Suburbs", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)


