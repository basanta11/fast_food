import streamlit as st
import osmnx as ox
import pandas as pd
import geopandas as gpd

# Title
st.title("Fast Food Locations in Sydney")

# Set up OSMnx
ox.settings.log_console = False

# Tags for fast food places
tags = {"amenity": "fast_food", "name": ["McDonald's", "KFC", "Subway"]}

# Load data
with st.spinner("Fetching data from OpenStreetMap..."):
    gdf = ox.features_from_place("Sydney, Australia", tags=tags)

# Select columns
columns_to_keep = ['name', 'geometry', 'brand', 'branch', 'addr:street']
df = gdf[columns_to_keep]

brand = st.selectbox("Select a Brand", ["McDonald's", "KFC", "Subway"])


st.subheader("Adjust Weights for Location Score")

# Slider for density weight (0 to 10)
density_weight = st.slider("Density Weight", min_value=0, max_value=10, value=5)

# Automatically adjust distance weight so that total = 10
distance_weight = 10 - density_weight
st.write(f"Distance Weight: {distance_weight}")

test_filtered = df[df['brand'] == brand]

# Show Data
st.subheader("Fast Food Locations Data")
st.dataframe(test_filtered.head())

if test_filtered.crs.is_geographic:
    test_filtered = test_filtered.to_crs(epsg=3857)

# Step 2: Calculate centroids and store in a new column
test_filtered["geometry_centroid"] = test_filtered.geometry.centroid

# Optional Step 3: Reproject centroids back to lat/lon (EPSG:4326)
test_filtered["geometry_centroid_latlon"] = (
    test_filtered["geometry_centroid"].to_crs(epsg=4326)
)

sa2_gdf = gpd.read_file("shape_files/SA2_2021_AUST_SHP_GDA2020/SA2_2021_AUST_GDA2020.shp")





sa2_nsw = sa2_gdf[sa2_gdf['STE_NAME21'] == 'New South Wales']

sa2 = sa2_nsw[sa2_nsw['GCC_NAME21'] == 'Greater Sydney']

census_data = pd.read_csv('shape_files/2021Census_G01_NSW_SA2.csv')
sa2=sa2[['SA2_CODE21', 'SA2_NAME21', 'geometry']]
census = census_data[['SA2_CODE_2021', 'Tot_P_P']]
census['SA2_CODE21']= census['SA2_CODE_2021'].astype(str)

sa2 = sa2.merge(census, on='SA2_CODE21', how='left')


# Convert to projected CRS (e.g., Australian Albers EPSG:3577)
sa2 = sa2.to_crs(epsg=3577)

# Calculate area in square kilometers
sa2['area_km2'] = sa2['geometry'].area / 1e6
sa2['pop_density'] = sa2['Tot_P_P'] / sa2['area_km2']
# Step 0: Make sure both GeoDataFrames are in the same projected CRS
if test_filtered.crs != sa2.crs:
    sa2 = sa2.to_crs(test_filtered.crs)

# Step 1: Define function to get closest polygon (row) from sa2
def get_nearest_suburb(point, suburbs_gdf):
    distances = suburbs_gdf.geometry.distance(point)
    return suburbs_gdf.loc[distances.idxmin()]

# Step 2: Apply the function to each centroid in test_filtered
# You can choose what attribute to extract (e.g., 'suburb_name', 'LGA_CODE', etc.)
test_filtered['nearest_suburb'] = test_filtered['geometry_centroid'].apply(
    lambda pt: get_nearest_suburb(pt, sa2)['SA2_NAME21']
)
suburb_list=sa2
fast_food=test_filtered[['name','geometry','nearest_suburb','geometry_centroid']]

# Step 1: Count how many times each suburb appears
suburb_counts = fast_food["nearest_suburb"].value_counts().reset_index()

# Step 2: Rename columns for clarity
suburb_counts.columns = ["suburb", "fast_food_count"]


suburb_list = suburb_list.merge(suburb_counts, left_on='SA2_NAME21', right_on='suburb', how="left")

# Step 3: Fill missing counts with 0 (for suburbs with no fast food matches)
suburb_list["fast_food_count"] = suburb_list["fast_food_count"].fillna(0).astype(int)


import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Project both to the same projected CRS (for distance accuracy)
suburb_list_proj = suburb_list.to_crs(epsg=3857).copy()
fast_food = fast_food.to_crs(suburb_list_proj.crs)

# Step 2: Calculate centroid for each SA2 area
suburb_list_proj['centroid'] = suburb_list_proj.geometry.centroid

# Step 3: Calculate minimum distance from each centroid to existing fast food venues
suburb_list_proj['min_dist_to_fastfood_km'] = suburb_list_proj['centroid'].apply(
    lambda x: fast_food.distance(x).min() / 1000  # convert to kilometers
)

# Step 2: Calculate area in square kilometers
suburb_list_proj.loc[:, 'area_km2'] = suburb_list_proj['geometry'].area / 1_000_000

# Step 3: Calculate population density
suburb_list_proj.loc[:, 'pop_density'] = suburb_list_proj['Tot_P_P'] / suburb_list_proj['area_km2']


new_list = suburb_list_proj


scaler = MinMaxScaler()
new_list[['norm_density', 'norm_dist']] = scaler.fit_transform(
    new_list[['pop_density', 'min_dist_to_fastfood_km']]
)

# Step 5: Create a composite score (weights: 0.7 for density, 0.3 for distance)
new_list['location_score'] = (
    new_list['norm_density'] * (density_weight/10) + new_list['norm_dist'] * (distance_weight/10)
)
POP_DENSITY_THRESHOLD = 1000
DISTANCE_TO_MCD_THRESHOLD_KM = 3.0

# Step 2: Filter the suburbs that meet both conditions
candidate_suburbs = new_list[
    (new_list['pop_density'] > POP_DENSITY_THRESHOLD) &
    (new_list['min_dist_to_fastfood_km'] > DISTANCE_TO_MCD_THRESHOLD_KM)
]

# Step 3: Preview the top candidates
top_areas = candidate_suburbs.sort_values(by='pop_density', ascending=False).head(3)
st.dataframe(top_areas[['SA2_NAME21', 'pop_density', 'min_dist_to_fastfood_km', 'location_score','fast_food_count']])


import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text

# --- Title ---
st.title("Top 3 Suggested Locations for a New Fast Food Venue")

# --- Load or use existing GeoDataFrames (replace with your actual loading logic) ---
# Example placeholders (you should load your real data here)
# fast_food = gpd.read_file("fast_food.geojson")
# suburb_list = gpd.read_file("suburbs.geojson")
# top_areas = suburb_list.sort_values("location_score", ascending=False).head(3)

# Dummy placeholders for testing
# st.write("Please replace dummy GeoDataFrames with your actual data")

# Reproject all to EPSG:3857
fast_food_wm = fast_food.to_crs(epsg=3857)
suburb_list_wm = suburb_list.to_crs(epsg=3857)
top_areas_wm = top_areas.to_crs(epsg=3857)

# Combine for zoom bounds
combined = pd.concat([fast_food_wm[['geometry']], top_areas_wm[['geometry']]], ignore_index=True)
minx, miny, maxx, maxy = combined.total_bounds
zoom_margin = 5000
xlim = (minx - zoom_margin, maxx + zoom_margin)
ylim = (miny - zoom_margin, maxy + zoom_margin)

# Plotting
fig, ax = plt.subplots(figsize=(12, 10))

# Plot fast food venues
fast_food_wm.plot(ax=ax, color='blue', markersize=5)

# Plot top 3 suggested SA2 locations
top_areas_wm.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)

# Annotate
texts = []
for idx, row in top_areas_wm.iterrows():
    texts.append(
        plt.text(
            row.geometry.centroid.x + 5000,
            row.geometry.centroid.y + 5000,
            row['SA2_NAME21'],
            fontsize=14, color='red'
        )
    )

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

# Zoom and basemap
ax.set_xlim(xlim)
ax.set_ylim(ylim)
cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

# Legend
legend_elements = [
    Patch(facecolor='white', edgecolor='grey', label='Suburbs'),
    Line2D([0], [0], marker='o', color='w', label='Fast Food Venue', markerfacecolor='blue', markersize=6),
    Line2D([0], [0], marker='s', color='r', label='Top 3 Suggested Areas', markerfacecolor='none', markersize=10)
]
ax.legend(handles=legend_elements)

# Title
ax.set_title("Top 3 Suggested Locations for New Fast Food Venue", fontsize=14)
plt.tight_layout()

# --- Show in Streamlit ---
st.pyplot(fig)

