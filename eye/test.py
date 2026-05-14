"""
test.py — Verifica se todas as dependências do ambiente estão instaladas e funcionando.
Execute dentro do container para validar o ambiente.
"""

import sys
import importlib

REQUIRED = [
    ("numpy",                        "numpy"),
    ("pandas",                       "pandas"),
    ("matplotlib",                   "matplotlib"),
    ("PIL",                          "Pillow"),
    ("cv2",                          "opencv-python-headless"),
    ("tqdm",                         "tqdm"),
    ("torch",                        "torch"),
    ("torchvision",                  "torchvision"),
    ("segmentation_models_pytorch",  "segmentation-models-pytorch"),
    ("albumentations",               "albumentations"),
    ("jupyterlab",                   "jupyterlab"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def check_import(module_name, package_name):
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "n/a")
        print(f"  {GREEN}✔{RESET}  {package_name:<35} versão {version}")
        return True
    except ImportError as e:
        print(f"  {RED}✘{RESET}  {package_name:<35} FALHOU → {e}")
        return False


def check_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = "GPU (CUDA)" if cuda else "CPU apenas"
        print(f"\n  {YELLOW}⚙{RESET}  PyTorch rodando em: {device}")
        print(f"        Versão CUDA do torch: {torch.version.cuda or 'N/A'}")
        t = torch.tensor([1.0, 2.0, 3.0])
        print(f"        Tensor de teste: {t.tolist()} — OK")
    except Exception as e:
        print(f"  {RED}Erro ao testar PyTorch:{RESET} {e}")


def check_opencv():
    try:
        import cv2
        import numpy as np
        img  = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (100, 100), "shape inesperado"
        print(f"        OpenCV conversão BGR→Gray: OK (shape {gray.shape})")
    except Exception as e:
        print(f"  {RED}Erro ao testar OpenCV:{RESET} {e}")


def check_segmentation():
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                         in_channels=3, classes=1)
        print(f"        U-Net (resnet18, sem pesos pré-treinados): construída OK")
    except Exception as e:
        print(f"  {RED}Erro ao testar segmentation_models_pytorch:{RESET} {e}")


def main():
    print(f"\n{'='*55}")
    print(f"  🐳  Verificação do ambiente Docker")
    print(f"  Python {sys.version.split()[0]}  |  {sys.platform}")
    print(f"{'='*55}\n")

    print("[ Verificando imports ]\n")
    results = [check_import(mod, pkg) for mod, pkg in REQUIRED]

    print("\n[ Testes funcionais ]\n")
    check_torch()
    check_opencv()
    check_segmentation()

    total  = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'='*55}")
    if failed == 0:
        print(f"  {GREEN}✔  Todos os {total} pacotes OK! Ambiente pronto.{RESET}")
    else:
        print(f"  {RED}✘  {failed}/{total} pacotes com falha.{RESET}")
    print(f"{'='*55}\n")
    sys.exit(0 if failed == 0 else 1)

def func():
    import matplotlib.pyplot as plt
    a = 2
    b = 37
    x=[]
    y=[]
    i=0
    while i >= 0:
        y.append(a*i + b  -0.4*i**2)
        x.append(i)
        i += 1
        if (y[i-1] > 1000 or y[i-1] < 0):
            break
    print(y)
    plt.plot(x, y)
    plt.savefig("test.png")


if __name__ == "__main__":
    # main()
    func()