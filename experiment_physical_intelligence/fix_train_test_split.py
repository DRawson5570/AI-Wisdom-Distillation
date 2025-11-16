#!/usr/bin/env python3
"""
Create proper train/test split with NO overlap.
"""

import json
import random
from failure_scenarios import load_scenarios
from pathlib import Path

# Load scenarios with same shuffle as evaluation
scenarios = load_scenarios("failure_scenarios_dataset.json")
random.seed(42)
random.shuffle(scenarios)

# Split: first 30 for training, next 19 for test
train_scenarios = scenarios[:30]
test_scenarios = scenarios[30:49]

train_ids = sorted([s.scenario_id for s in train_scenarios])
test_ids = sorted([s.scenario_id for s in test_scenarios])

print(f"Training IDs: {train_ids}")
print(f"Test IDs: {test_ids}")
print(f"\nOverlap check: {set(train_ids) & set(test_ids)}")

# Save the split for reference
split_info = {
    "train_ids": train_ids,
    "test_ids": test_ids,
    "train_count": len(train_ids),
    "test_count": len(test_ids)
}

with open("student_training/train_test_split.json", "w") as f:
    json.dump(split_info, f, indent=2)

print(f"\n✅ Saved split info to student_training/train_test_split.json")
