# NL2CAD 完整训练+评估+报告生成脚本
# 在独立的PowerShell窗口中运行，不依赖Cursor

param(
    [switch]$SkipTraining = $false
)

# 设置环境
$env:PYTHONIOENCODING = "utf-8"
$ProjectDir = "C:\Users\86185\Desktop\NL2CAD"
$ReportDir = "C:\Users\86185\Desktop\NL2CAD\repot"
$Python = "D:\conda\envs\NL2CAD\python.exe"

# 切换到项目目录
Set-Location $ProjectDir

function Write-Banner {
    param([string]$Text, [string]$Color = "Cyan")
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor $Color
    Write-Host $Text -ForegroundColor $Color
    Write-Host ("=" * 60) -ForegroundColor $Color
    Write-Host ""
}

function Get-Timestamp {
    return (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

# 开始
Write-Banner "NL2CAD 完整训练流程" "Cyan"
Write-Host "开始时间: $(Get-Timestamp)" -ForegroundColor Yellow
Write-Host "项目目录: $ProjectDir" -ForegroundColor Yellow
Write-Host "输出目录: $ReportDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "[提示] 此脚本在独立窗口运行，关闭Cursor不会影响训练!" -ForegroundColor Green
Write-Host "[提示] 按 Ctrl+C 可以手动停止" -ForegroundColor Yellow
Write-Host ""

# ========== 步骤1: 训练 ==========
if (-not $SkipTraining) {
    Write-Banner "步骤 1/3: 模型训练" "Yellow"
    
    $TrainingLog = Join-Path $ReportDir "training_log.txt"
    Write-Host "训练日志: $TrainingLog" -ForegroundColor Gray
    
    & $Python Cad_VLM/train.py -c Cad_VLM/config/trainer.yaml 2>&1 | Tee-Object -FilePath $TrainingLog
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "训练失败! 退出码: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "训练完成! 时间: $(Get-Timestamp)" -ForegroundColor Green
}
else {
    Write-Host "跳过训练步骤 (使用 -SkipTraining 参数)" -ForegroundColor Yellow
}

# ========== 步骤2: 评估 ==========
Write-Banner "步骤 2/3: 模型评估" "Yellow"

# 查找最新的检查点
$LatestCheckpoint = Get-ChildItem $ReportDir -Recurse -Filter "*_model.pth" | 
Sort-Object LastWriteTime -Descending | 
Select-Object -First 1

if ($LatestCheckpoint) {
    Write-Host "使用检查点: $($LatestCheckpoint.FullName)" -ForegroundColor Gray
    
    # 运行评估 (如果evaluate.py存在)
    if (Test-Path "Cad_VLM/evaluate.py") {
        & $Python Cad_VLM/evaluate.py -c Cad_VLM/config/inference.yaml --checkpoint $($LatestCheckpoint.FullName) 2>&1 | Tee-Object -FilePath (Join-Path $ReportDir "evaluation_log.txt")
    }
    else {
        Write-Host "评估脚本不存在，跳过评估" -ForegroundColor Yellow
    }
}
else {
    Write-Host "未找到检查点文件，跳过评估" -ForegroundColor Yellow
}

# ========== 步骤3: 生成报告 ==========
Write-Banner "步骤 3/3: 生成报告和图表" "Yellow"

if (Test-Path "generate_report.py") {
    & $Python generate_report.py 2>&1
    Write-Host "报告生成完成!" -ForegroundColor Green
}
else {
    Write-Host "报告生成脚本不存在" -ForegroundColor Yellow
}

# ========== 完成 ==========
Write-Banner "全部完成!" "Green"
Write-Host "结束时间: $(Get-Timestamp)" -ForegroundColor Yellow
Write-Host ""
Write-Host "输出文件位置: $ReportDir" -ForegroundColor Cyan
Write-Host ""

# 列出生成的文件
Write-Host "生成的文件:" -ForegroundColor Yellow
Get-ChildItem $ReportDir -Recurse -File | 
Where-Object { $_.Extension -in ".json", ".pth", ".txt", ".png", ".csv" } |
ForEach-Object { Write-Host "  - $($_.FullName)" -ForegroundColor Gray }

Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
