# trainer.py

import torch
import time
from tqdm import tqdm # <--- 1. IMPORTAR A BIBLIOTECA

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train(self):
        """Executa o ciclo completo de treinamento por todas as épocas."""
        print("Iniciando o treinamento...")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate_one_epoch()

            # Guardamos o histórico
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Fim da Época {epoch+1} -> Treino Loss: {train_loss:.4f}, Treino Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

        total_time = time.time() - start_time
        print(f"Treinamento concluído em {total_time//60:.0f}m {total_time%60:.0f}s")

    def _train_one_epoch(self, epoch_num):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        progress_bar = tqdm(self.train_loader, desc=f"Época {epoch_num+1}/{self.num_epochs} [Treino]")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def _validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        progress_bar = tqdm(self.val_loader, desc="Validando")

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def save_model(self, path):
        print(f"Salvando modelo em {path}...")
        torch.save(self.model.state_dict(), path)
