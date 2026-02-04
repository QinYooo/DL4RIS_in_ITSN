# ================= Configuration Area =================
# Output Filename
$OUTPUT_FILE = "train_dataset_mp_raw.pt"

# Total number of samples to generate
$NUM_SAMPLES = 10000

# Number of parallel workers (Recommended: CPU Cores - 2)
# Set to $null to use all available cores
$NUM_WORKERS = 50 

# Random Seed (Change this for test set, e.g., 2024)
$SEED = 42
# ======================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   ITSN Dataset Parallel Generation (Windows)"
Write-Host "============================================"
Write-Host "Target Samples: $NUM_SAMPLES"
Write-Host "Num Workers:    $NUM_WORKERS"
Write-Host "Random Seed:    $SEED"
Write-Host "============================================"

# Get the directory where this script is located
$ScriptDir = Split-Path $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Construct Python arguments
$PythonArgs = @(
    "generate_dataset.py",
    "--num_samples", $NUM_SAMPLES,
    "--seed", $SEED,
    "--output_name", $OUTPUT_FILE
)

if ($NUM_WORKERS) {
    $PythonArgs += "--num_workers"
    $PythonArgs += $NUM_WORKERS
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Using Python: $pythonVersion" -ForegroundColor Gray
}
catch {
    Write-Error "Python not found! Please check your PATH."
    exit
}

# Execute Python Script
python @PythonArgs

Write-Host "`nScript execution finished." -ForegroundColor Green
Pause