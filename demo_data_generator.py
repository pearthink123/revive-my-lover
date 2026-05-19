
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Generate demo data
random.seed(42)
np.random.seed(42)

now = datetime.now()
results = []

# Simulate 48 hours
for i in range(96):  # Every 30 minutes
    tick_time = now + timedelta(minutes=i * 30)
    hour = tick_time.hour
    
    # Simulate Poisson probability growth
    base_prob = 0.072
    growth = 0.08 * (i % 10)  # Reset every ~5 hours
    probability = min(0.95, base_prob + growth)
    
    # Simulate state based on hour
    if 0 <= hour < 7:
        state = "sleeping"
        utility = 0.01
    elif 9 <= hour < 17:
        state = "busy"
        utility = 0.15
    elif 18 <= hour < 22:
        state = "idle"
        utility = 0.65
    else:
        state = "away"
        utility = 0.35
    
    # Decide to send
    should_send = (probability > 0.5 and utility > 0.4 and random.random() < 0.3)
    
    results.append({
        "time": tick_time.isoformat(),
        "hour": hour,
        "probability": round(probability, 3),
        "state": state,
        "utility": round(utility, 2),
        "should_send": should_send,
    })

# Save to JSON
with open("demo_data.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Generated {len(results)} demo data points")
