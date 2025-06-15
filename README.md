# 🌿 Detecção de Doenças em Plantas com PyTorch e Transfer Learning

Este projeto utiliza uma Rede Neural Convolucional (CNN) com a técnica de Aprendizagem por Transferência (Transfer Learning) para classificar e identificar 15 tipos diferentes de doenças (ou condições saudáveis) em folhas de plantas a partir de imagens.

O modelo é construído em PyTorch, utilizando a arquitetura **MobileNetV2** pré-treinada na base de dados ImageNet, o que garante alta acurácia e treinamento rápido.

## ✨ Funcionalidades

- **Classificação de 15 categorias** de folhas de plantas (pimentão, batata e tomate).
- **Alta Acurácia** graças ao uso de Transfer Learning.
- **Estrutura de Projeto Profissional**, separando configurações, dados, modelo e lógica de treinamento.
- **Scripts para o ciclo completo de Machine Learning:**
    1.  `main.py`: Treinamento do modelo.
    2.  `evaluate.py`: Avaliação detalhada da performance com estatísticas completas.
    3.  `predict.py`: Classificação de uma única imagem fornecida pelo usuário.
- **Geração de Relatórios**: Cria um relatório de performance (`evaluation_report.txt`) e uma Matriz de Confusão visual (`confusion_matrix.png`).

## 📈 Resultados do Modelo

Após o treinamento por 10 épocas, o modelo foi avaliado em um conjunto de validação com 5160 imagens, alcançando os seguintes resultados:

- **Acurácia Geral**: **`97.38%`**

### Relatório de Classificação Detalhado

| Classe (Doença) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Pepper__bell___Bacterial_spot | 1.00 | 1.00 | 1.00 |
| Pepper__bell___healthy | 1.00 | 1.00 | 1.00 |
| Potato___Early_blight | 0.99 | 0.99 | 0.99 |
| Potato___Late_blight | 0.99 | 0.98 | 0.99 |
| Potato___healthy | 1.00 | 0.98 | 0.99 |
| Tomato_Bacterial_spot | 0.98 | 0.98 | 0.98 |
| Tomato_Early_blight | 0.93 | 0.89 | 0.91 |
| Tomato_Late_blight | 0.95 | 0.97 | 0.96 |
| Tomato_Leaf_Mold | 0.96 | 0.99 | 0.98 |
| Tomato_Septoria_leaf_spot | 0.97 | 0.93 | 0.95 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 0.93 | 0.99 | 0.96 |
| Tomato__Target_Spot | 0.97 | 0.90 | 0.93 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 0.99 | 0.99 | 0.99 |
| Tomato__Tomato_mosaic_virus | 0.99 | 0.97 | 0.98 |
| Tomato_healthy | 0.98 | 1.00 | 0.99 |

### Matriz de Confusão

A matriz de confusão abaixo ilustra visualmente a performance do modelo, mostrando os acertos na diagonal principal e os erros de classificação fora dela.


`![Matriz de Confusão](caminho/para/sua/confusion_matrix.png)`

## 📊 Dataset

O projeto utiliza o dataset público **PlantVillage**, que contém imagens de folhas de plantas saudáveis e com diversas doenças.

- **Fonte**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease/)
- **Conteúdo**: Imagens divididas em 15 pastas, cada uma representando uma classe.

## 📂 Estrutura do Projeto

O repositório está organizado da seguinte forma para garantir modularidade e clareza:

```
/plant_disease_project/
├── data/
│   └── PlantVillage/      <-- O dataset deve ser colocado aqui
├── saved_models/          <-- Modelos treinados são salvos aqui
|
├── config.py              # Arquivo de configuração (parâmetros, caminhos)
├── dataset.py             # Lógica de carregamento e preparação dos dados
├── model.py               # Definição da arquitetura do modelo (MobileNetV2)
├── trainer.py             # Classe que gerencia o laço de treinamento e validação
├── main.py                # Ponto de entrada para iniciar o treinamento
├── evaluate.py            # Script para gerar estatísticas do modelo treinado
├── predict.py             # Script para fazer previsões em novas imagens
└── README.md              # Este arquivo
```

## 🚀 Setup e Instalação

Siga os passos abaixo para configurar e executar o projeto.

**1. Clone o Repositório** (ou prepare sua pasta de projeto)

**2. Crie a Estrutura de Pastas**
Certifique-se de que as pastas `data/` e `saved_models/` existem na raiz do projeto.

**3. Baixe e Posicione o Dataset**
Baixe o dataset do link do Kaggle, descompacte-o e mova o conteúdo (a pasta `PlantVillage` com as 15 subpastas de classes) para dentro da pasta `data/`. A estrutura final deve ser `data/PlantVillage/Pepper__bell___healthy/...`.

**4. Crie um Ambiente Virtual (Recomendado)**
```bash
python -m venv venv
source venv/bin/activate
```

**5. Instale as Dependências**
Crie um arquivo `requirements.txt` com o conteúdo abaixo:
```txt
torch
torchvision
torchaudio
numpy
matplotlib
scikit-learn
seaborn
tqdm
Pillow
```
E então instale tudo com um único comando:
```bash
pip install -r requirements.txt
```

## ⚙️ Como Usar

O fluxo de trabalho é dividido em três etapas principais. Execute os comandos a partir da pasta raiz do projeto.

### **Etapa 1: Treinar o Modelo**

Este comando inicia o processo completo de treinamento, que irá ler os dados, construir o modelo e treiná-lo por 10 épocas (ou o que estiver definido em `config.py`).

```bash
python main.py
```
- **Saídas**: Ao final, dois arquivos serão criados:
    1.  `saved_models/plant_disease_model.pth` (os pesos do modelo treinado)
    2.  `class_names.json` (a lista de classes para referência)

### **Etapa 2: Avaliar a Performance do Modelo**

Após o treinamento, execute este script para gerar um relatório detalhado sobre o quão bem o seu modelo performou no conjunto de validação.

```bash
python evaluate.py
```
- **Saídas**: Este script gera dois arquivos de relatório:
    1.  `evaluation_report.txt` (Acurácia geral, precisão, recall e F1-score para cada classe).
    2.  `confusion_matrix.png` (Um gráfico visual que mostra onde o modelo acertou e errou).

### **Etapa 3: Fazer uma Previsão em uma Nova Imagem**

Com um modelo treinado e avaliado, você pode usá-lo para classificar novas imagens.

```bash
python predict.py --image "caminho/para/sua/imagem.jpg"
```
- **Saída**: O script imprimirá no terminal o diagnóstico da imagem e a confiança da previsão.
  ```
  Diagnóstico: Tomato___Late_blight
  Confiança: 98.72%
  ```

## 🛠️ Tecnologias Utilizadas

- **PyTorch**: Framework principal de Deep Learning.
- **Torchvision**: Para carregar modelos pré-treinados e transformações de imagem.
- **Scikit-learn**: Para calcular métricas de avaliação (acurácia, matriz de confusão, etc.).
- **Seaborn & Matplotlib**: Para visualização de dados e plotagem da matriz de confusão.
- **NumPy**: Para manipulação numérica.
- **Tqdm**: Para criar barras de progresso informativas durante o treinamento.