import wandb
import time
import random

# Initialize
wandb.init(project="gpu-test", name="rtx4090-test")

# Simulate training
for epoch in range(10):
    # Fake metrics
    loss = 1.0 / (epoch + 1) + random.random() * 0.1
    
    # Log to W&B (uploads tiny amount of data)
    wandb.log({"loss": loss, "epoch": epoch})
    time.sleep(0.5)

print("✓ Check your dashboard at https://wandb.ai/")
wandb.finish()
