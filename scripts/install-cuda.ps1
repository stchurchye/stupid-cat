# 安装 CUDA 版 PyTorch（GTX 1060）。大文件约 2.5GB，临时目录放在 F 盘。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

$tmp = Join-Path $Root ".tmp"
$cache = Join-Path $Root ".pip-cache"
New-Item -ItemType Directory -Force -Path $tmp, $cache | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp
$env:PIP_CACHE_DIR = $cache

.\.venv\Scripts\Activate.ps1
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
