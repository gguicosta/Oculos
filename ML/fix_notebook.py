"""
fix_notebook.py
Corrige Teste2.ipynb:
  1. Injeta célula de warmup da GPU (resolve travamento inicial do contexto CUDA)
  2. Corrige erro "Numpy is not available" (TF.to_tensor → conversão manual via np.array)
  3. Limpa outputs de erro das células anteriores
"""
import json, copy, sys

NB_PATH = "/workspaces/coding/Teste2.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ──────────────────────────────────────────────────────────────
# 1. Célula de warmup GPU — inserida após a célula de configs
# ──────────────────────────────────────────────────────────────
WARMUP_MARKDOWN = {
    "cell_type": "markdown",
    "id": "warmup_md_01",
    "metadata": {},
    "source": [
        "## 1.5 — Warmup da GPU (obrigatório antes do treino)\n",
        "Em dev containers com WSL2, a **primeira operação CUDA demora 60–90 s** para inicializar o contexto do driver.  \n",
        "Execute esta célula **uma única vez** e aguarde a confirmação antes de continuar."
    ]
}

WARMUP_CODE = {
    "cell_type": "code",
    "execution_count": None,
    "id": "gpu_warmup_01",
    "metadata": {},
    "outputs": [],
    "source": [
        "import time\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    print('⏳ Inicializando contexto CUDA — pode levar até 90 s na 1ª vez...')\n",
        "    t0 = time.time()\n",
        "\n",
        "    # Força a inicialização completa do driver\n",
        "    _w = torch.zeros(1, device='cuda')\n",
        "    torch.cuda.synchronize()   # garante que o driver respondeu\n",
        "    del _w\n",
        "    torch.cuda.empty_cache()\n",
        "\n",
        "    elapsed = time.time() - t0\n",
        "    vram_free = (torch.cuda.get_device_properties(0).total_memory\n",
        "                 - torch.cuda.memory_allocated(0)) / 1e9\n",
        "    print(f'✅ GPU pronta em {elapsed:.1f}s')\n",
        "    print(f'   VRAM livre: {vram_free:.2f} GB')\n",
        "    print(f'   Agora é seguro executar as células de treino.')\n",
        "else:\n",
        "    print('⚠️  GPU não disponível — rodando na CPU (treinamento muito lento).')"
    ]
}

# Encontra o índice da célula de configurações (id 03598584)
config_idx = next(
    (i for i, c in enumerate(cells) if c.get("id") == "03598584"), None
)

if config_idx is None:
    # fallback: procura pelo conteúdo
    config_idx = next(
        (i for i, c in enumerate(cells)
         if c["cell_type"] == "code" and any("DEVICE" in s for s in c.get("source", []))),
        None
    )

if config_idx is not None:
    # Limpa outputs de erro da célula de configurações
    cells[config_idx]["outputs"] = []
    cells[config_idx]["execution_count"] = None
    # Insere warmup depois da célula de configs
    cells.insert(config_idx + 1, WARMUP_CODE)
    cells.insert(config_idx + 1, WARMUP_MARKDOWN)
    print(f"✅ Células de warmup inseridas após índice {config_idx}")
else:
    print("⚠️  Célula de configs não encontrada — inserindo no início")
    cells.insert(2, WARMUP_CODE)
    cells.insert(2, WARMUP_MARKDOWN)

# ──────────────────────────────────────────────────────────────
# 2. Corrige células com TF.to_tensor → conversão manual via np
#    (evita RuntimeError: Numpy is not available)
# ──────────────────────────────────────────────────────────────
OLD_TO_TENSOR_IMAGE = "        image = TF.to_tensor(image)  # [1, H, W], valores 0-1\n"
NEW_TO_TENSOR_IMAGE = (
    "        # Conversao manual: evita bug 'Numpy not available' do TF.to_tensor\n"
    "        arr = np.array(image, dtype=np.float32) / 255.0\n"
    "        image = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]\n"
)

# Para segment_image: TF.to_tensor dentro de segment_image
OLD_TO_TENSOR_SEG = "    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)  # [1,1,H,W]\n"
NEW_TO_TENSOR_SEG = (
    "    arr = np.array(img_resized, dtype=np.float32) / 255.0\n"
    "    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]\n"
)

fixed_to_tensor = 0
fixed_error_outputs = 0

for cell in cells:
    if cell["cell_type"] != "code":
        continue

    # Corrige TF.to_tensor no Dataset
    new_source = []
    changed = False
    for line in cell.get("source", []):
        if line == OLD_TO_TENSOR_IMAGE:
            new_source.append(NEW_TO_TENSOR_IMAGE)
            changed = True
        elif line == OLD_TO_TENSOR_SEG:
            new_source.append(NEW_TO_TENSOR_SEG)
            changed = True
        else:
            new_source.append(line)
    if changed:
        cell["source"] = new_source
        fixed_to_tensor += 1

    # Remove outputs que contêm erros
    if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
        cell["outputs"] = []
        cell["execution_count"] = None
        fixed_error_outputs += 1

print(f"✅ TF.to_tensor corrigidos: {fixed_to_tensor}")
print(f"✅ Outputs de erro limpos: {fixed_error_outputs}")

# ──────────────────────────────────────────────────────────────
# 3. Salva
# ──────────────────────────────────────────────────────────────
nb["cells"] = cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Notebook salvo: {NB_PATH}")
print("   Recarregue o arquivo no VS Code (Ctrl+Shift+P → 'Revert File')")
print("   e execute as células em ordem a partir do Warmup da GPU.")
