$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
cargo build --release --bin nb-pick
$dist = Join-Path $root 'dist\NBPick'
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Get-ChildItem -LiteralPath $dist -File |
    Where-Object { $_.Name -notin @('班里挑一.exe', 'DirectML.dll') } |
    Remove-Item -Force
Copy-Item target\release\nb-pick.exe (Join-Path $dist '班里挑一.exe') -Force
Copy-Item target\release\DirectML.dll $dist -Force -ErrorAction SilentlyContinue
Copy-Item target\debug\DirectML.dll $dist -Force -ErrorAction SilentlyContinue
Copy-Item assets $dist -Recurse -Force
Remove-Item -LiteralPath (Join-Path $dist 'assets\models\face_detection_yunet_2023mar.onnx') -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $dist 'assets\models\LICENSE-YuNet') -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $dist '使用说明.md') -Force -ErrorAction SilentlyContinue
$redist = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC' -Directory |
    Where-Object Name -NE 'onecore' | Sort-Object Name -Descending | Select-Object -First 1
if ($redist) {
    Get-ChildItem (Join-Path $redist.FullName 'x64\Microsoft.VC*.CRT') -Filter '*.dll' -ErrorAction SilentlyContinue |
        Copy-Item -Destination $dist -Force
}
Write-Host "Portable application: $dist"
Pop-Location
