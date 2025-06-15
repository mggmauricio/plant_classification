import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/PlantVillage"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 15
MODEL_PATH = "saved_models/plant_disease_model.pth"
