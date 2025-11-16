#!/usr/bin/env python3
"""
Generate expanded curriculum with more training examples.
Uses data augmentation on existing scenarios.
"""

import json
import random
from pathlib import Path
from failure_scenarios import load_scenarios

def augment_scenario(scenario, variation_id):
    """Create variations of a scenario by modifying parameters"""
    # Vary grip force
    base_force = scenario.initial_strategy.get("grip_force_newtons", 5.0)
    force_variations = [base_force * 0.7, base_force, base_force * 1.3, base_force * 1.5]
    
    # Vary movement speed
    speed_variations = ["slow", "moderate", "fast"]
    
    # Pick variation based on ID
    force = force_variations[variation_id % len(force_variations)]
    speed = speed_variations[(variation_id // 2) % len(speed_variations)]
    
    # Create modified strategy
    modified_strategy = scenario.initial_strategy.copy()
    modified_strategy["grip_force_newtons"] = force
    modified_strategy["movement_speed"] = speed
    modified_strategy["reasoning"] = f"Variation {variation_id} - testing different parameters"
    
    return modified_strategy

def load_curriculum_template():
    """Load existing curriculum as template"""
    with open("student_training/curriculum.json") as f:
        return json.load(f)

def generate_expanded_curriculum(target_size=200):
    """Generate expanded curriculum by augmenting existing examples"""
    
    # Load train/test split
    print(f"Loading train/test split...")
    with open("student_training/train_test_split.json") as f:
        split_info = json.load(f)
        allowed_train_ids = set(split_info["train_ids"])
    print(f"✅ Loaded split: {len(allowed_train_ids)} training scenarios allowed")
    
    # Load all scenarios
    print(f"\nLoading all scenarios...")
    all_scenarios = load_scenarios("failure_scenarios_dataset.json")
    
    # Filter to only training scenarios
    train_scenarios_only = [s for s in all_scenarios if s.scenario_id in allowed_train_ids]
    
    print(f"✅ Filtered to {len(train_scenarios_only)} training scenarios (no test overlap)")
    
    # Create scenario lookup by ID (only training scenarios)
    scenario_map = {s.scenario_id: s for s in train_scenarios_only}
    
    # Generate variations for each training scenario
    expanded = []
    examples_per_scenario = target_size // len(train_scenarios_only)
    
    print(f"\nGenerating {target_size} examples ({examples_per_scenario} variations per scenario)...")
    
    for scenario in train_scenarios_only:
        # Create variations
        for var_id in range(examples_per_scenario):
            # Create augmented version
            modified_strategy = augment_scenario(scenario, var_id)
            
            # Build new prompt
            prompt = f"""Object: {scenario.object_name}
Action: {scenario.attempted_action}
Strategy: {json.dumps(modified_strategy, indent=2)}
Environment: {', '.join(scenario.environmental_factors) if scenario.environmental_factors else 'None'}

Predict failure using physics principles."""
            
            # Build completion with physics reasoning
            completion = f"""PREDICTION: YES
FAILURE_MODE: {scenario.failure_mode.value}
REASONING: {scenario.failure_description}

Physics Principle: {scenario.physical_principle_violated}"""
            
            new_example = {
                "prompt": prompt,
                "completion": completion,
                "scenario_id": scenario.scenario_id,
                "variation_id": var_id
            }
            
            expanded.append(new_example)
    
    # Shuffle to mix variations
    random.seed(42)
    random.shuffle(expanded)
    
    print(f"✅ Generated {len(expanded)} total examples")
    
    # Save
    output_file = "student_training/curriculum_expanded.json"
    with open(output_file, 'w') as f:
        json.dump(expanded, f, indent=2)
    
    print(f"💾 Saved to {output_file}")
    
    # Stats
    failure_modes = {}
    for ex in expanded:
        completion = ex["completion"]
        if "FAILURE_MODE:" in completion:
            mode = completion.split("FAILURE_MODE:")[1].split("\n")[0].strip().lower()
            failure_modes[mode] = failure_modes.get(mode, 0) + 1
    
    print(f"\nFailure mode distribution:")
    for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count}")
    
    return expanded

if __name__ == "__main__":
    expanded_curriculum = generate_expanded_curriculum(target_size=200)
    print(f"\n✅ Complete! Generated {len(expanded_curriculum)} training examples")
