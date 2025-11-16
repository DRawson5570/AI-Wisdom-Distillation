# Physical Intelligence Distillation

**The Common-Sense Butler: Distilling Physical Intuition into an Embodied Agent**

## Hypothesis

The abstract, common-sense physical reasoning of a massive Vision-Language Model (the "Teacher") can be distilled into a compact, real-time policy network (the "Student") on a robot, enabling the robot to handle novel objects and situations with a degree of "physical intuition" it could never learn on its own.

## Experimental Design

### Phase 1: Teacher's Education (LRL in Physics)
Teacher VLM learns to manipulate grocery items by analyzing failures:
- **Task**: Unpack groceries (milk, eggs, bread, apples, etc.)
- **Oracle**: Physical outcomes (crushed, dropped, slipped, tipped)
- **Reflection**: Physics journal analyzing violations of physical principles
- **Distillation**: "Physical Common Sense Manual" synthesized every 100 failures

### Phase 2: Student's Embodiment (LoRA Transfer)
- **Curriculum**: Teacher generates 10k synthetic training examples with reasoning
- **Transfer**: Fine-tune lightweight vision model with LoRA
- **Result**: Student robot with compiled physical intuition as reflexes

### Phase 3: Evaluation
- **Zero-shot generalization** to novel objects
- **Graceful failure handling** and recovery
- **Speed vs accuracy** vs Teacher real-time inference

## Project Structure

```
experiment_physical_intelligence/
├── physical_taxonomy.py          # Object categories, grip strategies, failure modes
├── failure_scenarios.py          # Generate 500+ failure scenarios
├── teacher_training.py           # LRL loop for Teacher VLM
├── curriculum_generation.py      # Teacher creates synthetic training data
├── student_training.py           # LoRA fine-tuning pipeline
├── evaluation.py                 # Test framework
├── teacher_logs/                 # Teacher journal & policy checkpoints
├── student_training/             # Training data & model checkpoints
├── evaluation/                   # Test results & metrics
└── models/                       # Saved models
```

## Quick Start

### 1. Generate Failure Scenarios
```bash
python failure_scenarios.py
```
Creates 500 diverse manipulation failure scenarios covering:
- Crushing (fragile items)
- Dropping (heavy items, low friction)
- Slipping (smooth surfaces)
- Tipping (liquid containers)
- Rolling (spherical objects)
- Sloshing (liquid dynamics)
- Collision (motion planning)

### 2. Train Teacher VLM
```bash
python teacher_training.py
```
Teacher analyzes failures and develops physical intuition through LRL loop.

### 3. Generate Curriculum
```bash
python curriculum_generation.py
```
Teacher creates synthetic training data with reasoning traces.

### 4. Train Student Model
```bash
python student_training.py
```
Fine-tune compact vision model with LoRA on Teacher's curriculum.

### 5. Evaluate
```bash
python evaluation.py
```
Test on novel objects, measure generalization and safety.

## Key Concepts

### Object Categories
- **Rigid**: Maintains shape (bottles, cans)
- **Fragile**: Breaks under pressure (eggs, glass)
- **Deformable**: Changes shape (bread, bags)
- **Rolling**: Spherical/cylindrical (fruits, cans)
- **Liquid-filled**: Contains liquid (milk, juice)

### Grip Strategies
- **Power Grip**: Full force, secure (heavy objects)
- **Cage Grip**: Distributed force (fragile items)
- **Surface Grip**: Low pressure, wide contact (deformable)
- **Precision Grip**: Fingertip control (small objects)
- **Stabilized Grip**: Anti-rotation (liquid containers)

### Physical Common Sense Principles
1. **Triage by Property**: Classify before acting
2. **Grip Selection**: Match strategy to object category
3. **Failure Prediction**: Anticipate risks from sensor data
4. **Safe Default**: Lower to stable surface on anomaly

## Expected Results

- **Student surpasses Teacher** in speed and consistency
- **Zero-shot generalization** to novel objects in same category
- **Graceful failure recovery** using learned heuristics
- **Verifiably safer** through auditable reasoning traces

## Why This is Profound

This experiment demonstrates:
- Abstract "common sense" can be distilled into embodied "intuition"
- Direct attack on the Sim-to-Real gap through principle transfer
- Psycho-epistemological transfer for the physical world
- Path to robots that work in messy, unpredictable environments
