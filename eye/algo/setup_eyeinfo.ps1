# ============================================================
#  setup_eyeinfo.ps1
#  Clona o repositório eyeinfo e baixa os dados via DVC
#  Execute em PowerShell a partir da pasta do projeto
# ============================================================

$REPO_URL  = "https://github.com/fabricionarcizo/eyeinfo.git"
$DEST_DIR  = "data\eyeinfo"

Write-Host "=== EyeInfo Dataset Setup ===" -ForegroundColor Cyan

# 1. Instalar DVC (se ausente)
if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    Write-Host "[1/4] Instalando DVC..." -ForegroundColor Yellow
    pip install "dvc[gdrive]" --quiet
} else {
    Write-Host "[1/4] DVC já instalado: $(dvc --version)" -ForegroundColor Green
}

# 2. Clonar o repositório (só código + .dvc pointers, sem dados)
if (-not (Test-Path $DEST_DIR)) {
    Write-Host "[2/4] Clonando repositório..." -ForegroundColor Yellow
    git clone $REPO_URL $DEST_DIR
} else {
    Write-Host "[2/4] Repositório já existente — atualizando..." -ForegroundColor Green
    Set-Location $DEST_DIR
    git pull
    Set-Location ..\..\
}

# 3. Baixar os dados via DVC (abre navegador para auth Google Drive)
Write-Host "[3/4] Baixando dados via DVC (pode abrir navegador para login Google)..." -ForegroundColor Yellow
Set-Location $DEST_DIR
dvc pull
Set-Location ..\..\

# 4. Verificar estrutura
Write-Host "[4/4] Verificando estrutura dos dados..." -ForegroundColor Yellow
$expected = @("01_dataset", "02_eye_feature", "03_metadata")
foreach ($folder in $expected) {
    $path = Join-Path $DEST_DIR "data\$folder"
    if (Test-Path $path) {
        $count = (Get-ChildItem $path -Recurse -File).Count
        Write-Host "  OK  $folder  ($count arquivos)" -ForegroundColor Green
    } else {
        Write-Host "  FALTANDO  $path" -ForegroundColor Red
    }
}

Write-Host "`n=== Setup concluido! ===" -ForegroundColor Cyan
Write-Host "Dados em: $((Resolve-Path $DEST_DIR).Path)"
