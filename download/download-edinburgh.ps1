$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataRoot = Join-Path $projectRoot "data\edinburgh"
$archiveRoot = Join-Path $projectRoot "data\edinburgh_archives"

New-Item -ItemType Directory -Force $dataRoot | Out-Null
New-Item -ItemType Directory -Force $archiveRoot | Out-Null

$files = @(
    @{
        Name = "clean_trainset_28spk_wav.zip"
        Url  = "https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download"
    },
    @{
        Name = "noisy_trainset_28spk_wav.zip"
        Url  = "https://datashare.ed.ac.uk/bitstreams/ecb5a102-bb00-46d3-8af5-40c79823b837/download"
    },
    @{
        Name = "clean_testset_wav.zip"
        Url  = "https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download"
    },
    @{
        Name = "noisy_testset_wav.zip"
        Url  = "https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download"
    }
)

foreach ($file in $files) {
    $archivePath = Join-Path $archiveRoot $file.Name
    $targetDir = Join-Path $dataRoot ([IO.Path]::GetFileNameWithoutExtension($file.Name))

    if (-not (Test-Path $targetDir)) {
        Write-Host "Downloading $($file.Name) ..."
        & curl.exe -L --fail --retry 3 --retry-delay 5 -o $archivePath $file.Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $($file.Name)"
        }

        Write-Host "Extracting $($file.Name) ..."
        & tar.exe -xf $archivePath -C $dataRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Extraction failed: $($file.Name). Try Expand-Archive manually."
        }
    }
    else {
        Write-Host "Already extracted: $targetDir"
    }
}

$required = @(
    "clean_trainset_28spk_wav",
    "noisy_trainset_28spk_wav",
    "clean_testset_wav",
    "noisy_testset_wav"
)

foreach ($name in $required) {
    $path = Join-Path $dataRoot $name
    if (-not (Test-Path $path)) {
        throw "Expected directory not found: $path"
    }
}

Write-Host "Edinburgh Noisy Speech Database is ready at: $dataRoot"
