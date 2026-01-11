# NL2CAD 独立训练脚本
# 此脚本可以在独立的PowerShell窗口中运行，不依赖Cursor

# 设置环境
$env:PYTHONIOENCODING = "utf-8"

# 切换到项目目录
Set-Location "C:\Users\86185\Desktop\NL2CAD"

# 激活conda环境并运行训练
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NL2CAD 训练启动中..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "训练日志将保存到: C:\Users\86185\Desktop\NL2CAD\repot\training_log.txt" -ForegroundColor Yellow
Write-Host ""
Write-Host "提示: 即使关闭Cursor，训练也会继续运行" -ForegroundColor Green
Write-Host "按 Ctrl+C 可以停止训练" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# 运行训练并保存日志
& D:\conda\envs\NL2CAD\python.exe Cad_VLM/train.py -c Cad_VLM/config/trainer.yaml 2>&1 | Tee-Object -FilePath "C:\Users\86185\Desktop\NL2CAD\repot\training_log.txt"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "训练完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# 等待用户按键
Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
