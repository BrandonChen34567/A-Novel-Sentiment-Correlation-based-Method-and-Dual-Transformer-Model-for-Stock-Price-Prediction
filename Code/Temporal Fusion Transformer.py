"""
Temporal Fusion Transformer (TFT) for Stock Price Prediction
"""

# ============================================================================
# IMPORTS
# ============================================================================

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE, MAE, SMAPE, PoissonLoss, QuantileLoss

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model hyperparameters
MAX_ENCODER_LENGTH = 5      # Past days to use for prediction
MAX_PREDICTION_LENGTH = 1     # Predict 1 day ahead
BATCH_SIZE = 64
HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 4
HIDDEN_CONTINUOUS_SIZE = 32
DROPOUT = 0.1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
GRADIENT_CLIP_VAL = 0.1
TEST_SIZE = 0.3

# Feature columns (modify as needed)
#This is just an example, you can modify it based on your dataset and features
FEATURE_COLS = [
    'Close', 
    'Amazon_polarity', 
    'Apple_polarity', 
    'BYD_polarity', 
    'GM_polarity', 
    'Ford_polarity'
]

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Add time and group identifiers
stock_data['date'] = stock_data.index
stock_data['group'] = "PEP"
stock_data["time_idx"] = range(len(stock_data))

# Create target variable (predict next day's close)
stock_data['Close_target'] = stock_data['Close'].shift(-1)
stock_data = stock_data.dropna()

# Split data
train_df, val_df = train_test_split(
    stock_data, 
    test_size=TEST_SIZE, 
    shuffle=False
)

# Scale features
scaler = MinMaxScaler()
train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
val_df[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])

# ============================================================================
# CREATE DATASETS AND DATALOADERS
# ============================================================================

# Create training dataset
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="Close",
    group_ids=["group"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=FEATURE_COLS,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Create validation dataset
validation = TimeSeriesDataSet.from_dataset(
    training, 
    val_df, 
    predict=True, 
    stop_randomization=True
)

# Create dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEAD_SIZE,
    hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
    dropout=DROPOUT,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=3,
)

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Callbacks
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=1e-4, 
    patience=10, 
    verbose=False, 
    mode="min"
)
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

# Trainer
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    gradient_clip_val=GRADIENT_CLIP_VAL,
    limit_train_batches=10,  # Remove this for full training
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

# ============================================================================
# MODEL TRAINING
# ============================================================================

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# ============================================================================
# LOAD BEST MODEL
# ============================================================================

best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# ============================================================================
# EVALUATION
# ============================================================================

# Make predictions
predictions = best_tft.predict(
    val_dataloader, 
    return_y=True, 
    trainer_kwargs=dict(accelerator="cpu")
)

# Unpack predictions and true labels
y_pred, y_true_tuple = predictions.output, predictions.y
y_true = y_true_tuple[0]

# Calculate metrics
mse = y_pred.sub(y_true).pow(2).mean()

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"MSE:   {mse.item():.6f}")                     # Mean Squared Error
print(f"RMSE:  {RMSE()(y_pred, y_true).item():.6f}")  # Root Mean Squared Error
print(f"MAE:   {MAE()(y_pred, y_true).item():.6f}")   # Mean Absolute Error
print(f"SMAPE: {SMAPE()(y_pred, y_true).item():.6f}") # Symmetric MAPE
print("=" * 50)