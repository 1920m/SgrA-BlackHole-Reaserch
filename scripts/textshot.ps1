# Render console/text output to a PNG "screenshot" (headless-safe).
# Usage: powershell -File scripts\textshot.ps1 -InFile out.txt -OutFile shot.png [-Title "ctest run"]
param(
    [Parameter(Mandatory=$true)][string]$InFile,
    [Parameter(Mandatory=$true)][string]$OutFile,
    [string]$Title = ""
)
Add-Type -AssemblyName System.Drawing

$lines = Get-Content -Path $InFile -Encoding UTF8
if ($null -eq $lines) { $lines = @("<empty output>") }
if ($lines -is [string]) { $lines = @($lines) }
if ($Title -ne "") { $lines = @("== $Title ==", "") + $lines }
# Cap extremely long captures so the PNG stays reasonable
if ($lines.Count -gt 300) {
    $lines = $lines[0..149] + @("", "... [$($lines.Count - 250) lines elided] ...", "") + $lines[($lines.Count-100)..($lines.Count-1)]
}

$font = New-Object System.Drawing.Font("Consolas", 11)
$measureBmp = New-Object System.Drawing.Bitmap(1,1)
$g0 = [System.Drawing.Graphics]::FromImage($measureBmp)
$maxW = 400
foreach ($l in $lines) {
    $sz = $g0.MeasureString(($l -replace "`t","    "), $font)
    if ($sz.Width -gt $maxW) { $maxW = [int]$sz.Width }
}
$g0.Dispose(); $measureBmp.Dispose()

$lineH = 18
$w = [Math]::Min($maxW + 30, 2200)
$h = $lines.Count * $lineH + 30

$bmp = New-Object System.Drawing.Bitmap($w, $h)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::FromArgb(255, 18, 18, 24))
$brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 214, 219, 223))
$green = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 120, 220, 120))
$red   = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 240, 110, 110))
$y = 15
foreach ($l in $lines) {
    $t = ($l -replace "`t","    ")
    $b = $brush
    if ($t -match "(?i)(passed|100% tests|OK\b|\bgreen\b)") { $b = $green }
    if ($t -match "(?i)(failed|error|\bred\b)")  { $b = $red }
    $g.DrawString($t, $font, $b, 12, $y)
    $y += $lineH
}
$g.Dispose()
$dir = Split-Path -Parent $OutFile
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
$bmp.Save($OutFile, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Output "textshot saved: $OutFile"
