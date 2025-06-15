# predict.py

import torch
import argparse
import json
from PIL import Image
from torchvision import transforms
import config
from model import create_model

def predict(image_path, model, class_names, device):
    """Faz uma previsão em uma única imagem."""
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Erro: O arquivo de imagem não foi encontrado em '{image_path}'")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score

def main():
    """Função principal para carregar o modelo e fazer a previsão."""
    parser = argparse.ArgumentParser(description="Classifica a doença de uma planta a partir de uma imagem.")
    parser.add_argument("--image", required=True, help="Caminho para a imagem de entrada.")
    args = parser.parse_args()

    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print("Erro: Arquivo 'class_names.json' não encontrado.")
        print("Por favor, execute o script de treinamento 'main.py' primeiro.")
        return

    model = create_model(len(class_names))

    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Erro: Modelo treinado não encontrado em '{config.MODEL_PATH}'")
        print("Por favor, execute o script de treinamento 'main.py' primeiro.")
        return

    model.to(config.DEVICE)

    predicted_disease, confidence = predict(args.image, model, class_names, config.DEVICE)

    if predicted_disease:
        print(f"Diagnóstico: {predicted_disease}")
        print(f"Confiança: {confidence*100:.2f}%")

if __name__ == '__main__':
    main()
