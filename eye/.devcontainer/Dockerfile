# Usa uma imagem robusta com PyTorch, Python e CUDA (suporte a GPU)
FROM python:3.13

# Instala utilitários de sistema essenciais (wget é exigido pelo Antigravity para instalar o servidor remoto)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Define a pasta de trabalho
WORKDIR /workspace

# Instala bibliotecas vitais para física, análise e comunicação de dados
RUN pip install --no-cache-dir \
    numpy scipy pandas matplotlib ipykernel
# websockets pyserial scikit-learn scikit-image