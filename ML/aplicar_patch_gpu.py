"""
Script para aplicar otimizações de GPU no Teste2.ipynb.
Execute com: python aplicar_patch_gpu.py
"""
import json, copy, shutil, pathlib

NB_PATH = pathlib.Path(__file__).parent / "Teste2.ipynb"
BACKUP  = NB_PATH.with_suffix(".ipynb.bak")

# ── Carregar notebook ──────────────────────────────────────────────────────────
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Backup antes de modificar
shutil.copy(NB_PATH, BACKUP)
print(f"Backup salvo em: {BACKUP}")

# ── Células novas ──────────────────────────────────────────────────────────────
CELL_IMPORTS = """\
import os
import matplotlib
matplotlib.rcParams['figure.dpi'] = 80  # evita outputs grandes no Jupyter
import glob
import base64
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.cuda.amp as amp          # Mixed Precision (AMP)
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm\
"""

CELL_CONFIG = """\

# ──── CONFIGURAÇÕES ────────────────────────────────────────────────────────────
BASE_DIR  = '/workspaces/coding/openEDS'   # pasta com S_0, S_1, ...
CSV_PATH  = '/workspaces/coding/dados_olhos (2).csv'
MODEL_PATH = '/workspaces/coding/unet_eye.pth'

IMG_SIZE   = (256, 256)   # resize para treino/inferência
N_CLASSES  = 4            # 0=fundo, 1=íris, 2=pupila, 3=esclera

# ──── OTIMIZAÇÕES DE GPU ───────────────────────────────────────────────────────
# Aumente BATCH_SIZE até encher a VRAM (comece com 16 ou 32).
# Se aparecer 'CUDA out of memory', reduza pela metade.
BATCH_SIZE = 16           # era 8 → dobra a ocupação da GPU
N_EPOCHS   = 10            # aumente para melhor desempenho
LR         = 1e-3
MAX_BATCHES_PER_EPOCH = None  # None = usa tudo; coloque ex: 200 para teste rápido

# Número de processos paralelos para carregar dados (libera a GPU de esperar a CPU)
# No Windows com containers, use 0 se der erro de multiprocessing.
NUM_WORKERS = 4

# Mixed Precision: True acelera ~2x em GPUs Ampere/Turing/Volta (RTX 20xx+)
USE_AMP = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM total:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'VRAM usada:  {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
    # Otimização cuDNN para convoluções com tamanho fixo
    torch.backends.cudnn.benchmark = True

CLASS_NAMES  = ['Fundo', 'Íris', 'Pupila', 'Esclera']
CLASS_COLORS = ['black', 'royalblue', 'gold', 'lightcoral']\
"""

CELL_DATASET_LOADER = """\
class OpenEDSDataset(Dataset):
    \"\"\"Dataset que emparelha PNG (imagem) com NPY (máscara) do OpenEDS.\"\"\"

    def __init__(self, base_path, img_size=(256, 256)):
        self.img_size = img_size
        png_files = sorted(glob.glob(os.path.join(base_path, 'S_*', '*.png')))
        self.pairs = []
        for png in png_files:
            npy = os.path.splitext(png)[0] + '.npy'
            if os.path.exists(npy):
                self.pairs.append((png, npy))
        print(f'Pares imagem+máscara encontrados: {len(self.pairs)}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        png_path, npy_path = self.pairs[idx]

        # Imagem em escala de cinza → tensor [1, H, W] float
        image = Image.open(png_path).convert('L')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = TF.to_tensor(image)  # [1, H, W], valores 0-1

        # Máscara inteira → tensor [H, W] long
        mask = np.load(npy_path).astype(np.int64)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


dataset = OpenEDSDataset(BASE_DIR, img_size=IMG_SIZE)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,                            # carregamento paralelo de dados
    pin_memory=True,                                    # transferência CPU→GPU mais rápida
    persistent_workers=(NUM_WORKERS > 0),              # mantém workers vivos entre épocas
    prefetch_factor=2 if NUM_WORKERS > 0 else None,    # pré-carrega próximos batches
)\
"""

CELL_UNET = """\
def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c,  out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    \"\"\"U-Net com encoder/decoder de 4 níveis para segmentação 1-canal → N classes.\"\"\"

    def __init__(self, in_channels=1, n_classes=4, base_f=64):  # base_f=64 → 4x mais parâmetros que 32
        super().__init__()
        f = base_f
        # Encoder
        self.enc1 = double_conv(in_channels, f)
        self.enc2 = double_conv(f,   f*2)
        self.enc3 = double_conv(f*2, f*4)
        self.enc4 = double_conv(f*4, f*8)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = double_conv(f*8, f*16)
        # Decoder
        self.up4   = nn.ConvTranspose2d(f*16, f*8,  kernel_size=2, stride=2)
        self.dec4  = double_conv(f*16, f*8)
        self.up3   = nn.ConvTranspose2d(f*8,  f*4,  kernel_size=2, stride=2)
        self.dec3  = double_conv(f*8,  f*4)
        self.up2   = nn.ConvTranspose2d(f*4,  f*2,  kernel_size=2, stride=2)
        self.dec2  = double_conv(f*4,  f*2)
        self.up1   = nn.ConvTranspose2d(f*2,  f,    kernel_size=2, stride=2)
        self.dec1  = double_conv(f*2,  f)
        self.out   = nn.Conv2d(f, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


model = UNet(in_channels=1, n_classes=N_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'U-Net inicializada — parâmetros treináveis: {total_params:,}')
if torch.cuda.is_available():
    print(f'VRAM ocupada pelo modelo: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB')\
"""

CELL_TRAIN = """\
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# GradScaler para Mixed Precision — evita underflow em FP16
scaler = amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())

history = []

print(f'Treinando por {N_EPOCHS} épocas no dispositivo: {DEVICE}')
print(f'Mixed Precision (AMP): {USE_AMP and torch.cuda.is_available()}')
print(f'Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}')
print('=' * 55)

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc=f'Época {epoch}/{N_EPOCHS}', leave=False)
    for batch_idx, (imgs, masks) in enumerate(pbar):
        if MAX_BATCHES_PER_EPOCH and batch_idx >= MAX_BATCHES_PER_EPOCH:
            break

        # non_blocking=True: transferência assíncrona CPU→GPU (requer pin_memory=True)
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        # set_to_none=True: mais rápido que zerar os gradientes
        optimizer.zero_grad(set_to_none=True)

        # Forward com autocast (FP16 onde possível)
        with amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
            outputs = model(imgs)          # [B, C, H, W]
            loss    = criterion(outputs, masks)

        # Backward com scaler (compensa a escala do FP16)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds    = outputs.detach().argmax(dim=1)
        correct += (preds == masks).sum().item()
        total   += masks.numel()

        # Mostra uso de VRAM no progresso
        vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        pbar.set_postfix(loss=f'{loss.item():.4f}', vram=f'{vram_mb:.0f}MB')

    scheduler.step()
    avg_loss = running_loss / (batch_idx + 1)
    acc      = 100.0 * correct / total
    history.append({'epoch': epoch, 'loss': avg_loss, 'acc': acc})
    vram_peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    print(f'Época {epoch:02d} | Loss: {avg_loss:.4f} | Pixel Acc: {acc:.2f}% | VRAM pico: {vram_peak:.0f}MB')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

print('=' * 55)
print('Treinamento concluído!')

# Salvar pesos
torch.save(model.state_dict(), MODEL_PATH)
print(f'Modelo salvo em: {MODEL_PATH}')\
"""

# ── Mapear células por ID ──────────────────────────────────────────────────────
# IDs extraídos do notebook original:
# e76d96cc = Imports
# 03598584 = Configurações
# 1dd33c79 = Dataset
# 8c5e9024 = UNet
# c60caf37 = Treinamento

PATCHES = {
    "e76d96cc": CELL_IMPORTS,
    "03598584": CELL_CONFIG,
    "1dd33c79": CELL_DATASET_LOADER,
    "8c5e9024": CELL_UNET,
    "c60caf37": CELL_TRAIN,
}

applied = []
for cell in nb["cells"]:
    cell_id = cell.get("id", "")
    if cell_id in PATCHES and cell["cell_type"] == "code":
        new_source = PATCHES[cell_id]
        # Converte string para lista de linhas (formato nbformat)
        lines = new_source.splitlines(keepends=True)
        # Garante que a última linha não tem \n extra
        if lines:
            lines[-1] = lines[-1].rstrip('\n')
        cell["source"] = lines
        cell["outputs"] = []
        cell["execution_count"] = None
        applied.append(cell_id)
        print(f"  [OK] Patch aplicado na celula {cell_id}")

# -- Salvar notebook modificado ------------------------------------------------
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n{'='*55}")
print(f"Notebook salvo: {NB_PATH}")
print(f"Células modificadas: {len(applied)}/{len(PATCHES)}")
print(f"Backup disponível em: {BACKUP}")
