$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

$blockedExtensions = @(
    ".safetensors", ".pt", ".pth", ".bin", ".ckpt",
    ".mp4", ".pdf", ".tgz", ".tar", ".zip", ".parquet", ".arrow"
)

$blocked = Get-ChildItem -LiteralPath $root -Recurse -File -Force |
    Where-Object { $blockedExtensions -contains $_.Extension.ToLowerInvariant() }

if ($blocked) {
    Write-Host "Blocked release files found:" -ForegroundColor Red
    $blocked | Select-Object FullName, Length | Format-Table -AutoSize
    exit 1
}

$large = Get-ChildItem -LiteralPath $root -Recurse -File -Force |
    Where-Object { $_.Length -gt 50MB }

if ($large) {
    Write-Host "Files larger than 50MB found:" -ForegroundColor Red
    $large | Select-Object FullName, Length | Format-Table -AutoSize
    exit 1
}

$required = @(
    "README.md",
    "README_CN.md",
    "ENVIRONMENT_REQUIREMENTS.txt",
    "requirements.txt",
    "requirements/dreamzero.txt",
    "requirements/lingbot-va.txt",
    "requirements/act-pi05.txt",
    "projects/dreamzero/train_wam_sft.sh",
    "projects/dreamzero/run_grpo3500_compare.sh",
    "projects/dreamzero/groot/vla/grpo_simple.py",
    "projects/lingbot-va/wan_va/train.py",
    "projects/lingbot-va/wan_va/grpo.py",
    "projects/lingbot-va/script/run_lingbot_grpo_ablation.sh",
    "projects/act-grpo-datatest/resources/lerobot/src/lerobot/utils/act_grpo.py",
    "projects/act-grpo-datatest/scripts/train_act_bc_dataset.sh",
    "projects/act-grpo-datatest/scripts/train_act_grpo_dataset.sh",
    "projects/act-grpo-datatest/scripts/train_pi05_grpo_dataset.sh",
    "projects/pi05/scripts/train_pi05_grpo_dataset.sh"
)

$missing = foreach ($path in $required) {
    $full = Join-Path $root $path
    if (-not (Test-Path -LiteralPath $full)) {
        $path
    }
}

if ($missing) {
    Write-Host "Required files missing:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}

$count = (Get-ChildItem -LiteralPath $root -Recurse -File -Force | Measure-Object).Count
$size = (Get-ChildItem -LiteralPath $root -Recurse -File -Force | Measure-Object Length -Sum).Sum

Write-Host "Release check passed." -ForegroundColor Green
Write-Host ("Files: {0}" -f $count)
Write-Host ("Total size: {0:N2} MB" -f ($size / 1MB))
