# Fast Food Franchise Location Prediction in Sydney

[![View Interactive Streamlit App](https://img.shields.io/badge/Streamlit-Demo-orange?logo=streamlit)](https://fastfood-tzahejbu6bkyngfgylecgx.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12xvCIPyhirh0Q1bNXtO_e6jN5wZmzaBi?usp=sharing)

[![View on Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/basantadahal/franchise-new-location-predictor)
  


## Project Overview
This project uses geospatial analysis and machine learning to identify optimal locations for new fast food franchise outlets in Greater Sydney. By combining open geospatial data, census statistics, and predictive modeling, the workflow helps franchise owners and analysts make data-driven decisions about expansion.

## Online Demos
- **Streamlit App:** [View Interactive App](https://fastfood-tzahejbu6bkyngfgylecgx.streamlit.app/)  
  Launch the interactive dashboard to explore and visualize recommended locations.
- **Google Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/12xvCIPyhirh0Q1bNXtO_e6jN5wZmzaBi?usp=sharing)  
  Run the full analysis in your browser without local setup.

## Key Features
- **Geospatial Data Integration:** Uses OSMnx to download and analyze fast food locations (McDonald's, KFC, Subway) in Sydney.
- **Census Data Analysis:** Integrates 2021 Australian Census data to calculate population density for each suburb (SA2 region).
- **Spatial Matching:** Matches existing fast food venues to their nearest suburb using centroid calculations and spatial joins.
- **Coverage Visualization:** Plots existing venues and their coverage buffers to highlight underserved areas.
- **Location Scoring:** Calculates a composite score for each suburb based on normalized population density and distance to the nearest fast food venue.
- **Candidate Selection:** Filters and ranks suburbs to recommend top locations for new outlets, based on density and distance thresholds.
- **Machine Learning Prediction:** Trains a Random Forest classifier to predict the probability that a suburb is suitable for a new outlet, using features like population density and proximity to competitors.
- **Map Visualization:** Visualizes recommended locations and existing venues on interactive maps with annotations and basemaps.

## Workflow Summary
1. **Import Libraries:** Load geospatial, data science, and visualization libraries.
2. **Download Fast Food Data:** Use OSMnx to fetch locations of major fast food brands in Sydney.
3. **Prepare Data:** Select relevant columns, filter by brand, and calculate centroids.
4. **Load Shapefiles & Census Data:** Read suburb boundaries and census population data, merge them, and calculate population density.
5. **Spatial Analysis:** Match venues to suburbs, count venues per suburb, and visualize coverage.
6. **Scoring & Ranking:** Normalize metrics, compute location scores, and filter top candidate suburbs.
7. **Machine Learning:** Train/test a classifier to predict suitability for new outlets and rank unserved suburbs.
8. **Visualization:** Plot results and recommendations on maps for easy interpretation.

## How to Run
1. Clone this repository and ensure you have Python 3.10+ installed.
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or place the required shapefiles and census CSVs in the `shape_files` directory.
4. Run the notebook (`main.ipynb`) or the Streamlit app (`main.py`) to explore results and visualizations.



## Customizable Variables
You can change the following variables in the notebook or Streamlit app to see different outputs:
- `brand`: Select which fast food franchise to analyze (e.g., "McDonald's", "KFC", "Subway").
- `POP_DENSITY_THRESHOLD`: Minimum population density required for candidate suburbs.
- `DISTANCE_TO_MCD_THRESHOLD_KM`: Minimum distance from existing outlets for new recommendations.
- `density_weight` and `distance_weight`: Adjust the importance of population density vs. distance in the location score.
- Model parameters: Change features or classifier settings for machine learning predictions.

These variables allow you to tailor the analysis to different brands, thresholds, and scoring strategies.

## Data Sources
- [OpenStreetMap](https://www.openstreetmap.org/) (via OSMnx)
- [Australian Bureau of Statistics](https://www.abs.gov.au/) (2021 Census)

## File Structure
- `main.ipynb`: Jupyter notebook with step-by-step analysis and visualizations.
- `main.py`: Streamlit app for interactive exploration.
- `shape_files/`: Contains shapefiles and census data.
- `README.md`: Project documentation.

## Example Outputs
- Ranked list of top suburbs for new fast food outlets.
- Interactive maps showing existing coverage and recommended locations.
- Classification report and feature importance for predictive modeling.

## License
This project is for educational and research purposes. Please cite data sources appropriately if used for commercial work.

## Contact
For questions or collaboration, please contact the repository owner via GitHub.
