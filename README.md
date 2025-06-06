# LLT: Lume Labelling Tool

LLT é uma ferramenta gráfica interativa para rotulagem e segmentação de imagens e vídeos, com suporte a modelos de segmentação automática (SAM 2). Permite criar, editar, visualizar e salvar anotações de objetos em sequências de imagens, facilitando o trabalho de anotação para tarefas de visão computacional.

---

## Instalação

**Pré-requisitos:**
- CUDA 12.1 ([guia de instalação](https://github.com/LumeRobotics/docs/blob/main/Installations/instaling_CUDA_12.1.md))
- Python >= 3.10
- torch >= 2.5.1
- torchvision >= 0.20.1

**Passos:**

```bash
# Instale Python 3.10
cd /opt
sudo wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tar.xz
sudo tar -xvf Python-3.10.0.tar.xz
cd Python-3.10.0
sudo ./configure --enable-optimizations
sudo make altinstall
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
sudo rm Python-3.10.0.tar.xz

# Instale dependências do LLT
cd lume_labelling_tool
python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3.10 -m pip install -e .
python3.10 -m pip install opencv-python
python3.10 -m pip install pyqt6==6.9.0
sudo apt install libxcb-cursor0
pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ..
```

---

## Como usar

### 1. Inicie o programa

```bash
python3.10 labelling_tool.py
```

### 2. Janela de Configuração Inicial

- **Caminho de Entrada:** Selecione a pasta com as imagens ou frames do vídeo.
- **Caminho de Saída:** (opcional) Pasta para salvar resultados/conversões.
- **Converter imagens a partir do log:** (opcional) Permite converter imagens de logs.
- **Camera ID:** (opcional) Selecione o ID da câmera (se aplicável).
- **Selecionar arquivo de dados (opcional):** Após escolher o caminho de entrada, uma combobox aparece listando todos os arquivos `data_*.pkl` do diretório. Você pode escolher um arquivo de dados anterior para continuar a anotação. Se não escolher, o mais recente será usado automaticamente.

Clique em **Iniciar** para abrir a interface principal.

---

### 3. Interface Principal

- **Visualização da Imagem:** Área central para exibir a imagem/frame atual.
- **Inference (p):** Checkbox para ativar/desativar o modo de predição automática (atalho: `P`).
- **Show all objs (h):** Checkbox para mostrar todas as anotações de objetos no frame (atalho: `H`).
- **Object ID:** Combobox para selecionar o ID do objeto atual.
- **New Object:** Cria um novo objeto (ID sequencial).
- **Delete Object:** Remove o objeto atual e todas as suas marcações em todos os frames.
- **Frame Index:** Campo para navegar diretamente para um frame específico.
- **OK:** Confirma a navegação para o frame digitado.

---

### 4. Fluxo de Trabalho

- **Navegação entre frames:**  
  - `A` ou rolar para cima: frame anterior  
  - `D` ou rolar para baixo: próximo frame

- **Criação e edição de objetos:**  
  - Clique em "New Object" para criar um novo ID de objeto.
  - Clique na imagem para adicionar pontos positivos (esquerdo) ou remover pontos (direito).
  - Clique com o botão do meio para propagar a segmentação para o próximo frame.
  - Use a combobox para alternar entre IDs de objetos.

- **Predição automática:**  
  - Ative o modo "Inference" para que o modelo SAM 2 sugira máscaras automaticamente ao navegar entre frames.

- **Salvar e carregar anotações:**  
  - As anotações são salvas automaticamente em `data.pkl` no diretório de entrada.
  - Ao fechar o programa, uma cópia de backup é criada com timestamp (`data_YYYYMMDD_HHMMSS.pkl`).

- **Carregar anotações anteriores:**  
  - Na tela inicial, selecione um arquivo de dados anterior para continuar a anotação.

---

## Atalhos de Teclado

- `P`: Alterna o modo de predição automática (Inference)
- `H`: Alterna a visualização de todos os objetos (Show all objs)
- `A`: Frame anterior
- `D`: Próximo frame

---

## Formato dos Dados

- As anotações são salvas em arquivos `.pkl` (pickle), contendo:
  - Máscaras, contornos, bounding boxes, classe e ID de cada objeto por frame.
- Backups automáticos são criados ao fechar o programa.

---

## Requisitos

- Python >= 3.10
- CUDA 12.1 (para uso com GPU)
- torch >= 2.5.1
- torchvision >= 0.20.1
- opencv-python
- pyqt6==6.9.0
- Outros: tqdm, numpy, pillow, natsort, etc.

---

## Observações e Limitações

- O desempenho do modo de predição depende do hardware (GPU recomendada).
- O programa pode ser usado com imagens ou sequências de frames extraídas de vídeos.
- O formato de dados é binário (pickle); para exportar para outros formatos, será necessário um script adicional.
- O backup automático evita perda de dados em caso de falha.

---

## Erros Comuns

- **IndexError ao navegar:** Certifique-se de que o frame atual está dentro do intervalo de imagens carregadas.
- **Problemas de dependências:** Verifique se todas as bibliotecas estão instaladas corretamente e se o CUDA está configurado.

---

## Licença

Este projeto é baseado em [SAM 2](https://github.com/facebookresearch/sam2) da Meta AI, sob licença Apache 2.0.

---

Se precisar de exemplos de uso, dicas de troubleshooting ou quiser adicionar instruções para exportação/conversão de dados, posso complementar o README!
