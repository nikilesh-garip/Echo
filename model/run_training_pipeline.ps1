Write-Host "Starting UrbanSound8K Ingestion (This will take a while)..."
python ingest_real_datasets.py --download-urbansound

if ($LASTEXITCODE -ne 0) {
    Write-Error "Ingestion failed!"
    exit $LASTEXITCODE
}

Write-Host "Generating Metadata..."
python generate_real_metadata.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "Metadata generation failed!"
    exit $LASTEXITCODE
}

Write-Host "Starting Model Training..."
python train.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed!"
    exit $LASTEXITCODE
}

Write-Host "Pipeline completed successfully!"
