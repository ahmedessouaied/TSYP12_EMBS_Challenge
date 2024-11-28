from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import shutil
import os
import subprocess
import pandas as pd
from tempfile import NamedTemporaryFile
import mfcc
import egemaps
import boaw

app = FastAPI()

# Folder to store temporary files
TEMP_FOLDER = "temp_audio_files"
os.makedirs(TEMP_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_form():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_audio_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the audio file with the three scripts (mfcc, egemaps, boaw)
    mfcc_csv = mfcc.process_mfcc(temp_audio_path)
    egemaps_csv = egemaps.process_egemaps(temp_audio_path)
    boaw_csv = boaw.process_boaw(temp_audio_path)

    # Combine the CSVs into one final CSV
    final_csv = combine_csvs(mfcc_csv, egemaps_csv, boaw_csv)

    # Save all CSV files
    mfcc_csv_path = "mfcc_features.csv"
    egemaps_csv_path = "egemaps_features.csv"
    boaw_csv_path = "boaw_features.csv"
    final_csv_path = "final_features.csv"

    mfcc_csv.to_csv(mfcc_csv_path, index=False)
    egemaps_csv.to_csv(egemaps_csv_path, index=False)
    boaw_csv.to_csv(boaw_csv_path, index=False)
    final_csv.to_csv(final_csv_path, index=False)

    # Return the final CSV and individual CSVs as downloadable files
    return {
        "mfcc_csv": FileResponse(mfcc_csv_path),
        "egemaps_csv": FileResponse(egemaps_csv_path),
        "boaw_csv": FileResponse(boaw_csv_path),
        "final_csv": FileResponse(final_csv_path)
    }

def combine_csvs(mfcc_df, egemaps_df, boaw_df):
    # Merge the dataframes on common columns (you might need to adjust based on your data)
    combined_df = pd.concat([mfcc_df, egemaps_df, boaw_df], axis=1)
    return combined_df
