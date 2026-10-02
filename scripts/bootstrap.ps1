param(
    [switch]$WithMl
)

$ErrorActionPreference = "Stop"
$UvVersion = "0.12.9"
$PythonVersion = "3.12.14"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$UvDirectory = Join-Path $ProjectRoot ".tools\uv"
$UvExecutable = Join-Path $UvDirectory "uv.exe"

if (-not (Test-Path -LiteralPath $UvExecutable)) {
    $UvArchive = Join-Path ([System.IO.Path]::GetTempPath()) "my-project-1-uv-$UvVersion.zip"
    $UvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"

    New-Item -ItemType Directory -Force -Path $UvDirectory | Out-Null
    Write-Host "Downloading uv $UvVersion..."
    Invoke-WebRequest -Uri $UvUrl -OutFile $UvArchive
    Expand-Archive -LiteralPath $UvArchive -DestinationPath $UvDirectory -Force
    Remove-Item -LiteralPath $UvArchive -Force
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".python"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"

Push-Location $ProjectRoot
try {
    & $UvExecutable python install $PythonVersion --no-registry
    if ($LASTEXITCODE -ne 0) {
        throw "Python installation failed with exit code $LASTEXITCODE."
    }

    $SyncArguments = @("sync", "--locked", "--extra", "dev", "--extra", "tokenizer")
    if ($WithMl) {
        $SyncArguments += @("--extra", "ml")
    }

    & $UvExecutable @SyncArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency synchronization failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Environment ready: $ProjectRoot\.venv"
