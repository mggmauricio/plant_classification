# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import get_dataloaders
from model import create_model
from trainer import Trainer
import json

def main():
    print(f"Usando o dispositivo: {config.DEVICE}")

    print("Carregando dados...")
    train_loader, val_loader, class_names = get_dataloaders(
        config.DATA_DIR, config.IMAGE_SIZE, config.BATCH_SIZE
    )
    print("Classes:", class_names)

    print("Criando o modelo...")
    model = create_model(len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.LEARNING_RATE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS
    )

    trainer.train()

    trainer.save_model(config.MODEL_PATH)

    print("Salvando os nomes das classes em class_names.json...")
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    # -----------------------------------------------------------

if __name__ == '__main__':
    main()
