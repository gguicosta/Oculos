"""
inject_dvc_cells.py
Insere 4 células de DVC/EyeInfo no olhin.ipynb logo após a célula cell_install.
"""
import json, pathlib

nb_path = pathlib.Path(r"g:/Meu Drive/Dissertação/Dados/Triple X/olhin.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# ── Célula Markdown explicativa ───────────────────────────────────────────────
new_md = {
    "cell_type": "markdown",
    "id": "md_eyeinfo_dvc",
    "metadata": {"id": "md_eyeinfo_dvc"},
    "source": [
        "## 1.5 EyeInfo Dataset — download via DVC\n",
        "\n",
        "> O repositório **fabricionarcizo/eyeinfo** usa DVC com armazenamento no"
        " **Google Drive** (~285 MB).  \n",
        "> Execute esta célula **uma única vez** para clonar o repo e baixar os dados.\n",
        "> Quando solicitado, autorize o acesso ao Google Drive no navegador.\n",
        "\n",
        "| Pasta gerada | Conteúdo |\n",
        "|---|---|\n",
        "| `data/eyeinfo/data/01_dataset/` | Frames brutos do eye-tracker |\n",
        "| `data/eyeinfo/data/02_eye_feature/` | CSVs de features oculares (pupila, íris, CR) |\n",
        "| `data/eyeinfo/data/03_metadata/` | Metadados dos participantes |",
    ],
}

# ── Célula de código: clone + dvc pull ────────────────────────────────────────
new_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell_eyeinfo_dvc",
    "metadata": {"id": "cell_eyeinfo_dvc"},
    "outputs": [],
    "source": [
        "# ══════════════════════════════════════════════════════════════════════════════\n",
        "#  EyeInfo Dataset — clone + DVC pull\n",
        "#  Ref: https://github.com/fabricionarcizo/eyeinfo\n",
        "# ══════════════════════════════════════════════════════════════════════════════\n",
        "import subprocess, sys, pathlib\n",
        "\n",
        "EYEINFO_REPO = 'https://github.com/fabricionarcizo/eyeinfo.git'\n",
        "EYEINFO_DIR  = pathlib.Path('data/eyeinfo')\n",
        "EYEINFO_DATA = EYEINFO_DIR / 'data'\n",
        "\n",
        "# 1) Instalar DVC com suporte a Google Drive\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'dvc[gdrive]', '--quiet'], check=True)\n",
        "print('DVC instalado ✓')\n",
        "\n",
        "# 2) Clonar repositório (apenas código + ponteiros .dvc)\n",
        "if not EYEINFO_DIR.exists():\n",
        "    subprocess.run(['git', 'clone', EYEINFO_REPO, str(EYEINFO_DIR)], check=True)\n",
        "    print('Repositório clonado ✓')\n",
        "else:\n",
        "    subprocess.run(['git', '-C', str(EYEINFO_DIR), 'pull'], check=True)\n",
        "    print('Repositório atualizado ✓')\n",
        "\n",
        "# 3) Baixar dados via DVC (abre navegador para autenticação Google)\n",
        "print('Iniciando dvc pull — autorize o acesso ao Google Drive se solicitado...')\n",
        "result = subprocess.run(['dvc', 'pull'], cwd=str(EYEINFO_DIR),\n",
        "                        capture_output=True, text=True)\n",
        "print(result.stdout)\n",
        "if result.returncode != 0:\n",
        "    print('[AVISO] dvc pull retornou erro:', result.stderr)\n",
        "else:\n",
        "    print('Dados EyeInfo baixados com sucesso ✓')\n",
        "\n",
        "# 4) Verificar estrutura\n",
        "print('\\n── Estrutura do dataset EyeInfo:')\n",
        "for folder_name in ['01_dataset', '02_eye_feature', '03_metadata']:\n",
        "    folder = EYEINFO_DATA / folder_name\n",
        "    if folder.exists():\n",
        "        n = len(list(folder.rglob('*')))\n",
        "        status = 'OK   '\n",
        "        print(f'  [{status}]  {folder_name}  ({n} itens)')\n",
        "    else:\n",
        "        print(f'  [FALTANDO]  {folder_name}')\n",
    ],
}

# ── Célula Markdown: schema EyeInfo ──────────────────────────────────────────
new_md2 = {
    "cell_type": "markdown",
    "id": "md_eyeinfo_read",
    "metadata": {"id": "md_eyeinfo_read"},
    "source": [
        "### Leitura dos CSVs de features EyeInfo\n",
        "\n",
        "Após o `dvc pull`, os CSVs ficam em `data/eyeinfo/data/02_eye_feature/`.  \n",
        "As colunas principais são:\n",
        "\n",
        "| Coluna | Descrição |\n",
        "|---|---|\n",
        "| `frame` | Número do frame |\n",
        "| `target_id` | ID do alvo visualizado |\n",
        "| `timestamp` | Timestamp (ms) |\n",
        "| `pupil_center_x/y` | Centro da pupila (px) |\n",
        "| `iris_major/minor` | Semi-eixos da íris (px) |\n",
        "| `iris_angle` | Ângulo do elipsoide da íris |\n",
        "| `cr1_x/y … cr4_x/y` | Reflexos corneais 1–4 |",
    ],
}

# ── Célula de código: ler CSVs ────────────────────────────────────────────────
new_code2 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell_eyeinfo_read",
    "metadata": {"id": "cell_eyeinfo_read"},
    "outputs": [],
    "source": [
        "# ── Leitura dos CSVs de features oculares do EyeInfo ─────────────────────────\n",
        "import pandas as pd, pathlib\n",
        "\n",
        "EYEINFO_FEAT = pathlib.Path('data/eyeinfo/data/02_eye_feature')\n",
        "\n",
        "if not EYEINFO_FEAT.exists():\n",
        "    print('[AVISO] Diretório de features não encontrado.'\n",
        "          ' Execute a célula de dvc pull acima primeiro.')\n",
        "else:\n",
        "    csv_files = sorted(EYEINFO_FEAT.rglob('*.csv'))\n",
        "    print(f'Arquivos CSV encontrados: {len(csv_files)}')\n",
        "\n",
        "    # Carrega e concatena todos os CSVs na pasta\n",
        "    dfs = []\n",
        "    for f in csv_files:\n",
        "        try:\n",
        "            tmp = pd.read_csv(f)\n",
        "            tmp['source_file'] = f.name\n",
        "            dfs.append(tmp)\n",
        "        except Exception as e:\n",
        "            print(f'[AVISO] {f.name}: {e}')\n",
        "\n",
        "    if dfs:\n",
        "        df_eyeinfo = pd.concat(dfs, ignore_index=True)\n",
        "        print(f'Total de registros EyeInfo : {len(df_eyeinfo)}')\n",
        "        print(f'Colunas                    : {list(df_eyeinfo.columns)}')\n",
        "        display(df_eyeinfo.head(5))\n",
        "    else:\n",
        "        print('Nenhum CSV carregado.')\n",
    ],
}

# ── Localizar posição de inserção ─────────────────────────────────────────────
cells = nb["cells"]
insert_pos = None
for i, c in enumerate(cells):
    if c.get("id") == "cell_install":
        insert_pos = i + 1
        break

if insert_pos is None:
    print("ERRO: célula cell_install não encontrada!")
else:
    cells.insert(insert_pos,     new_md)
    cells.insert(insert_pos + 1, new_code)
    cells.insert(insert_pos + 2, new_md2)
    cells.insert(insert_pos + 3, new_code2)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Células inseridas na posição {insert_pos}.")
    print(f"Total de células no notebook: {len(cells)}")
