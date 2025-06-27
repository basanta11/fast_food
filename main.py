import streamlit as st
import osmnx as ox
import pandas as pd
import geopandas as gpd
import pyproj
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
st.set_page_config(page_title="Fast Food Optimizer",layout="centered",  # ← Default (normal) layout
    initial_sidebar_state="auto")

# --------------------- HEADER --------------------------
st.title("🍔 Franchise Fast Food Location Optimizer – Sydney")
st.markdown("This tool analyzes population density and other franchise location to recommend ideal locations for new venues in Greater Sydney.")
st.markdown("---")

# ------------------- LOAD & PREP DATA ------------------



# ----------------- WEIGHT SLIDERS ----------------------


col1, col2 = st.columns([1, 1])
with col1:
    with st.spinner("🔄 Fetching fast food data..."):
        ox.settings.log_console = False
        tags = {"amenity": "fast_food", "name": ["McDonald's", "KFC", "Subway"]}
        gdf = ox.features_from_place("Sydney, Australia", tags=tags)
        df = gdf[['name', 'geometry', 'brand', 'branch', 'addr:street']]
        df = df[df['brand'].notnull()]
        brand_list = df['brand'].value_counts().index.tolist()

    brand = st.selectbox("🏪 Select a Fast Food Brand", brand_list)
with col2:
    st.subheader("🎯 Adjust Weights for Location Score")
  
    density_weight = st.slider("🏙️ Population Density Weight", 0, 10, 7)

    distance_weight = 10 - density_weight
    st.metric("📍 Distance Weight", value=distance_weight)

# ------------------- FILTER DATA -----------------------

test_filtered = df[df['brand'] == brand]
if test_filtered.crs.is_geographic:
    test_filtered = test_filtered.to_crs(epsg=3857)

test_filtered["geometry_centroid"] = test_filtered.geometry.centroid
test_filtered["geometry_centroid_latlon"] = test_filtered["geometry_centroid"].to_crs(epsg=4326)

# ------------------ LOAD SHAPEFILES --------------------

sa2_gdf = gpd.read_file("shape_files/SA2_2021_AUST_SHP_GDA2020/SA2_2021_AUST_GDA2020.shp")
sa2_nsw = sa2_gdf[sa2_gdf['STE_NAME21'] == 'New South Wales']
sa2 = sa2_nsw[sa2_nsw['GCC_NAME21'] == 'Greater Sydney'][['SA2_CODE21', 'SA2_NAME21', 'geometry']]

census_data = pd.read_csv('shape_files/2021Census_G01_NSW_SA2.csv')
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

# ----------------- SHOW MAP + RECOMMENDATIONS -----------------------

st.markdown("## 📍 Suggested Locations for New Outlet")
st.dataframe(top_areas[['SA2_NAME21', 'pop_density', 'Nearest_franchise_location(in km)', 'location_score']])

with st.spinner("🔄 Rendering location map..."):
    fast_food_wm = fast_food.to_crs(epsg=3857)
    suburb_list_wm = suburb_list.to_crs(epsg=3857)
    top_areas_wm = top_areas.to_crs(epsg=3857)

    combined = pd.concat([fast_food_wm[['geometry']], top_areas_wm[['geometry']]], ignore_index=True)
    minx, miny, maxx, maxy = combined.total_bounds
    fig, ax = plt.subplots(figsize=(10, 8))
    fast_food_wm.plot(ax=ax, color='blue', markersize=5)
    top_areas_wm.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)

    texts = []
    for idx, row in top_areas_wm.iterrows():
        texts.append(
            plt.text(
                row.geometry.centroid.x + 1000,
                row.geometry.centroid.y + 1000,
                row['SA2_NAME21'],
                fontsize=14, color='red'
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
    ax.set_title("Top 3 Suggested Suburbs", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

# -------------------- INTERACTIVE MAP --------------------
suburb_list_proj_new=suburb_list_proj.copy()
st.markdown("## 🗺️ Interactive Map – Click to Get Suburb Recommendation")

with st.spinner("🔄 Loading interactive map..."):
    m = folium.Map(location=[-33.87, 151.21], zoom_start=10)
    for _, row in suburb_list_proj_new.iterrows():
        folium.GeoJson(row["geometry"],
                       name=row.get("SA2_NAME21", "Unknown"),
                       tooltip=row.get("SA2_NAME21", "Unknown"),
                        style_function=lambda x: {
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 1
                        }).add_to(m)

    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        click_point = Point(lon, lat)
        project = pyproj.Transformer.from_crs("EPSG:4326", suburb_list_proj_new.crs, always_xy=True).transform
        click_point_proj = transform(project, click_point)
        matched = suburb_list_proj_new[suburb_list_proj_new.contains(click_point_proj)]

        if not matched.empty:
            row = matched.iloc[0]
            st.success(f"📍 You clicked inside: {row['SA2_NAME21']}")
            st.write(f"**Location Score:** {row['location_score']:.2f}")
            stats = suburb_list_proj_new['location_score'].describe()

            low_thresh = stats['25%']     # ≈ 0.075
            med_thresh = stats['50%']     # ≈ 0.126
            high_thresh = stats['75%'] 
            if row['location_score'] >= high_thresh:
                st.success("✅ Excellent location to open a venue!")
            elif row['location_score'] >= med_thresh:
                st.info("🟡 Moderately suitable area.")
            elif row['location_score'] < low_thresh:
                st.info("🟡 Low suitable area.")
            else:
                st.warning("🔴 Not ideal based on the current data.")
             # Create new map with highlight
            m_highlight = folium.Map(location=[lat, lon], zoom_start=12)
            folium.Marker([lat, lon], tooltip=row['SA2_NAME21']).add_to(m_highlight)

            # Add highlighted polygon
            folium.GeoJson(
                row['geometry'],
                name="Selected Suburb",
                style_function=lambda x: {
                    'fillColor': 'yellow',
                    'color': 'red',
                    'weight': 3,
                    'fillOpacity': 0.3,
                },
                tooltip=row['SA2_NAME21']
            ).add_to(m_highlight)

            st.markdown("### 🔎 Highlighted Suburb")
            st_folium(m_highlight, width=700, height=500)

        else:
            st.warning("⚠️ You clicked outside the suburb boundaries.")
        
