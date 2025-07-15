pip install papermill
import os
import papermill as pm
import shutil

# === CONFIGURATION ===
NOTEBOOK_PATH = "main.ipynb"
OUTPUT_NOTEBOOK = "main_executed.ipynb"
DATA_DIR = "data"
KAGGLE_DATA_DIR = "kaggle_input"

# === COPY CSV FILES TO 'kaggle_input' ===
if not os.path.exists(KAGGLE_DATA_DIR):
    os.makedirs(KAGGLE_DATA_DIR)

for file in os.listdir(DATA_DIR):
    src = os.path.join(DATA_DIR, file)
    dst = os.path.join(KAGGLE_DATA_DIR, file)
    shutil.copy(src, dst)

print(f"✅ Copied data files to '{KAGGLE_DATA_DIR}/'")

# === RUN NOTEBOOK ===
pm.execute_notebook(
    NOTEBOOK_PATH,
    OUTPUT_NOTEBOOK,
    parameters={"DATA_DIR": KAGGLE_DATA_DIR},
    kernel_name="python3"
)

print(f"✅ Executed notebook saved as '{OUTPUT_NOTEBOOK}'")
