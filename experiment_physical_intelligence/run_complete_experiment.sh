#!/bin/bash
#
# COMPLETE PHYSICS INTELLIGENCE EXPERIMENT WORKFLOW
# 
# This script runs the entire experiment from start to finish:
# 1. Data preparation (train/test split, curriculum generation)
# 2. LoRA training
# 3. Comprehensive evaluation (baseline + LoRA)
# 4. Results reporting
#
# Usage: ./run_complete_experiment.sh
#

set -e  # Exit on error

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXPERIMENT_DIR"

echo "================================================================================"
echo "PHYSICS INTELLIGENCE EXPERIMENT - COMPLETE WORKFLOW"
echo "================================================================================"
echo ""
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Start Time: $(date)"
echo ""

# Create logs directory
mkdir -p experiment_logs
LOG_FILE="experiment_logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "📝 Logging to: $LOG_FILE"
echo ""

# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================
echo "================================================================================"
echo "STEP 1: DATA PREPARATION"
echo "================================================================================"
echo ""

echo "1.1: Creating train/test split..."
python3 fix_train_test_split.py
echo "✅ Train/test split created"
echo ""

echo "1.2: Generating success scenarios..."
python3 success_scenarios.py
echo "✅ Success scenarios generated"
echo ""

echo "1.3: Generating expanded training curriculum..."
python3 generate_more_curriculum.py
echo "✅ Training curriculum generated"
echo ""

# ============================================================================
# STEP 2: LORA TRAINING
# ============================================================================
echo "================================================================================"
echo "STEP 2: LORA TRAINING"
echo "================================================================================"
echo ""

echo "2.1: Preparing training data (JSONL format)..."
python3 lora_train.py --prepare
echo "✅ Training data prepared"
echo ""

echo "2.2: Training LoRA adapter..."
echo "   This may take 5-10 minutes..."
rm -rf student_training/lora_output
python3 lora_train.py --train --model Qwen/Qwen2.5-0.5B --epochs 3 --batch-size 1
echo "✅ LoRA training complete"
echo ""

# ============================================================================
# STEP 3: COMPREHENSIVE EVALUATION
# ============================================================================
echo "================================================================================"
echo "STEP 3: COMPREHENSIVE EVALUATION"
echo "================================================================================"
echo ""

echo "3.1: Running full evaluation (baseline + LoRA)..."
echo "   Testing on balanced dataset (failures + successes)..."
python3 comprehensive_evaluation.py
echo "✅ Evaluation complete"
echo ""

# ============================================================================
# STEP 4: RESULTS SUMMARY
# ============================================================================
echo "================================================================================"
echo "STEP 4: EXPERIMENT COMPLETE"
echo "================================================================================"
echo ""

echo "📊 Results Location:"
echo "   - Detailed results: evaluation_results/"
echo "   - Training logs: student_training/lora_output/"
echo "   - Experiment log: $LOG_FILE"
echo ""

echo "End Time: $(date)"
echo ""
echo "================================================================================"
echo "✅ EXPERIMENT COMPLETE"
echo "================================================================================"
