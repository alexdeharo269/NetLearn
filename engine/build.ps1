# build.ps1 — compile the five ESN/MC programs on Windows (MSYS2 ucrt64 + Eigen).
#   Usage:  powershell -ExecutionPolicy Bypass -File build.ps1
# Override the toolchain if needed:
#   $env:GPP           = "C:\msys64\ucrt64\bin\g++.exe"
#   $env:EIGEN_INCLUDE = "C:\path\to\eigen-master"
$ErrorActionPreference = "Stop"

$gpp = if ($env:GPP) { $env:GPP } else { "C:\msys64\ucrt64\bin\g++.exe" }
if (-not (Test-Path $gpp)) {
    $w = (Get-Command g++ -ErrorAction SilentlyContinue).Source
    if ($w) { $gpp = $w } else { throw "g++ not found. Install MSYS2 ucrt64 or set `$env:GPP." }
}
$eigen = if ($env:EIGEN_INCLUDE) { $env:EIGEN_INCLUDE } else { Join-Path $PSScriptRoot "eigen-master" }
if (-not (Test-Path (Join-Path $eigen "Eigen\Dense"))) {
    throw "Eigen not found at '$eigen'. Put the eigen-master folder next to these .cpp files, or set `$env:EIGEN_INCLUDE."
}

Write-Host "g++   : $gpp"
Write-Host "eigen : $eigen`n"
$flags = @("-std=c++17", "-O2", "-fopenmp", "-I$eigen", "-w")   # C++17 is required (Eigen+C++20 breaks LDLT)
foreach ($p in "esn_mc", "esn_surrogates", "esn_bio", "esn_ipc", "esn_trace") {
    Write-Host "building $p ..."
    & $gpp @flags (Join-Path $PSScriptRoot "$p.cpp") -o (Join-Path $PSScriptRoot "$p.exe")
    if ($LASTEXITCODE -ne 0) { throw "compile failed: $p" }
}
Write-Host "`ndone — five .exe built in $PSScriptRoot"
