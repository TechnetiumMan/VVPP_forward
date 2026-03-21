import os

class Config:
    # ---------------------------------------------------------
    # Global Hyperparameters
    # ---------------------------------------------------------
    
    # Training Parameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    
    # Dataset Parameters
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    NUM_WORKERS = 4
    
    # Model Architecture Parameters
    INPUT_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 64
    
    # Feature Extraction Parameters
    N_MELS = 64
    SAMPLE_RATE = 16000
    N_EIGENMODES = 64

# Create a global instance to be imported
cfg = Config()
