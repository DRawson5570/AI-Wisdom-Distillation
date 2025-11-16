#!/usr/bin/env python3
"""
Comprehensive Evaluation System for Physics Intelligence Experiment

This script runs complete evaluation of:
1. Baseline model (Ollama qwen2.5:1.5b) - binary and failure mode accuracy
2. LoRA model (Qwen2.5-0.5B + physics adapter) - binary and failure mode accuracy
3. Balanced test set (failures + successes)

Generates detailed logs, comparisons, and publication-ready results.
"""

import json
import random
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import requests

# Import local modules
from failure_scenarios import load_scenarios, FailureScenario
from success_scenarios import SuccessScenario

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Configuration
RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_MODEL = "qwen2.5:1.5b"
LORA_BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_ADAPTER_PATH = Path("student_training/lora_output")

@dataclass
class EvaluationConfig:
    """Configuration for evaluation run"""
    run_id: str
    timestamp: str
    test_failure_count: int
    test_success_count: int
    baseline_model: str
    lora_model: str
    random_seed: int

@dataclass
class PredictionResult:
    """Single prediction result"""
    scenario_id: int
    object_name: str
    action: str
    actual_outcome: str  # "failure" or "success"
    actual_failure_mode: str  # for failures, "none" for successes
    predicted_outcome: str
    predicted_failure_mode: str
    raw_response: str
    is_correct_binary: bool
    is_correct_failure_mode: bool  # Only relevant for failures
    confidence: float = 1.0

class BaselineEvaluator:
    """Evaluates baseline model via Ollama"""
    
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def predict(self, scenario, is_failure: bool) -> PredictionResult:
        """Generate prediction for a scenario"""
        # Build prompt
        if is_failure:
            prompt = self._build_failure_prompt(scenario)
            actual_failure_mode = scenario.failure_mode.value
        else:
            prompt = self._build_success_prompt(scenario)
            actual_failure_mode = "none"
        
        # Query model
        response = self._query_ollama(prompt)
        
        # Parse response
        predicted_outcome, predicted_failure_mode = self._parse_response(response)
        
        actual_outcome = "failure" if is_failure else "success"
        is_correct_binary = (predicted_outcome == actual_outcome)
        is_correct_failure_mode = (predicted_failure_mode.lower() == actual_failure_mode.lower()) if is_failure else True
        
        return PredictionResult(
            scenario_id=scenario.scenario_id,
            object_name=scenario.object_name,
            action=scenario.attempted_action,
            actual_outcome=actual_outcome,
            actual_failure_mode=actual_failure_mode,
            predicted_outcome=predicted_outcome,
            predicted_failure_mode=predicted_failure_mode,
            raw_response=response,
            is_correct_binary=is_correct_binary,
            is_correct_failure_mode=is_correct_failure_mode
        )
    
    def _build_failure_prompt(self, scenario: FailureScenario) -> str:
        return f"""Object: {scenario.object_name}
Action: {scenario.attempted_action}
Strategy: {json.dumps(scenario.initial_strategy, indent=2)}
Environment: {', '.join(scenario.environmental_factors) if scenario.environmental_factors else 'None'}

Predict if this manipulation will fail and identify the failure mode.

Response format:
PREDICTION: [YES/NO]
FAILURE_MODE: [crushing/dropping/slipping/tipping/rolling_away/sloshing/collision/none]
REASONING: [brief explanation]"""
    
    def _build_success_prompt(self, scenario: SuccessScenario) -> str:
        return f"""Object: {scenario.object_name}
Action: {scenario.attempted_action}
Strategy: {json.dumps(scenario.initial_strategy, indent=2)}
Environment: {', '.join(scenario.environmental_factors) if scenario.environmental_factors else 'None'}

Predict if this manipulation will fail and identify the failure mode.

Response format:
PREDICTION: [YES/NO]
FAILURE_MODE: [crushing/dropping/slipping/tipping/rolling_away/sloshing/collision/none]
REASONING: [brief explanation]"""
    
    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model response to extract prediction and failure mode"""
        response_upper = response.upper()
        response_lower = response.lower()
        
        # Check for YES/NO
        has_yes = "PREDICTION: YES" in response_upper or response_upper.startswith("YES")
        has_no = "PREDICTION: NO" in response_upper or response_upper.startswith("NO")
        
        # Narrative style detection
        has_will_not = "will not" in response_lower or "won't" in response_lower or "cannot" in response_lower
        
        # Predict outcome
        if has_yes or has_will_not:
            predicted_outcome = "failure"
        elif has_no:
            predicted_outcome = "success"
        else:
            # Default to failure if unclear
            predicted_outcome = "failure"
        
        # Extract failure mode
        failure_modes = ["crushing", "dropping", "slipping", "tipping", "rolling_away", "sloshing", "collision"]
        predicted_failure_mode = "none"
        
        if "FAILURE_MODE:" in response_upper:
            mode_text = response.split("FAILURE_MODE:")[1].split("\n")[0].strip().lower()
            for mode in failure_modes:
                if mode in mode_text:
                    predicted_failure_mode = mode
                    break
        else:
            # Check for mode keywords in text
            for mode in failure_modes:
                if mode in response_lower:
                    predicted_failure_mode = mode
                    break
        
        return predicted_outcome, predicted_failure_mode

class LoRAEvaluator:
    """Evaluates LoRA fine-tuned model"""
    
    def __init__(self, base_model: str, adapter_path: Path):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers/peft not available")
        
        self.model, self.tokenizer = self._load_model(base_model, adapter_path)
    
    def _load_model(self, base_model: str, adapter_path: Path):
        """Load LoRA model"""
        print(f"Loading LoRA model: {base_model} + {adapter_path}")
        
        # Check for checkpoints
        if not (adapter_path / "adapter_config.json").exists():
            checkpoints = sorted(adapter_path.glob("checkpoint-*"))
            if checkpoints:
                adapter_path = checkpoints[-1]
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(base, str(adapter_path))
        model.eval()
        
        return model, tokenizer
    
    def predict(self, scenario, is_failure: bool) -> PredictionResult:
        """Generate prediction"""
        # Build prompt (same format for both)
        if is_failure:
            prompt = self._build_prompt(scenario, scenario.attempted_action)
            actual_failure_mode = scenario.failure_mode.value
        else:
            prompt = self._build_prompt(scenario, scenario.attempted_action)
            actual_failure_mode = "none"
        
        # Generate
        response = self._generate(prompt)
        
        # Parse
        predicted_outcome, predicted_failure_mode = self._parse_response(response)
        
        actual_outcome = "failure" if is_failure else "success"
        is_correct_binary = (predicted_outcome == actual_outcome)
        is_correct_failure_mode = (predicted_failure_mode.lower() == actual_failure_mode.lower()) if is_failure else True
        
        return PredictionResult(
            scenario_id=scenario.scenario_id,
            object_name=scenario.object_name,
            action=scenario.attempted_action,
            actual_outcome=actual_outcome,
            actual_failure_mode=actual_failure_mode,
            predicted_outcome=predicted_outcome,
            predicted_failure_mode=predicted_failure_mode,
            raw_response=response,
            is_correct_binary=is_correct_binary,
            is_correct_failure_mode=is_correct_failure_mode
        )
    
    def _build_prompt(self, scenario, action: str) -> str:
        strategy = scenario.initial_strategy
        env = scenario.environmental_factors if hasattr(scenario, 'environmental_factors') else []
        
        return f"""Object: {scenario.object_name}
Action: {action}
Strategy: {json.dumps(strategy, indent=2)}
Environment: {', '.join(env) if env else 'None'}

Predict failure using physics principles.

PREDICTION:"""
    
    def _generate(self, prompt: str) -> str:
        """Generate response"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse response (same as baseline)"""
        response_upper = response.upper()
        response_lower = response.lower()
        
        has_yes = "YES" in response_upper or "PREDICTION: YES" in response_upper
        has_no = "NO" in response_upper or "PREDICTION: NO" in response_upper
        has_will_not = "will not" in response_lower or "won't" in response_lower
        
        if has_yes or has_will_not:
            predicted_outcome = "failure"
        elif has_no:
            predicted_outcome = "success"
        else:
            predicted_outcome = "failure"
        
        # Extract failure mode
        failure_modes = ["crushing", "dropping", "slipping", "tipping", "rolling_away", "sloshing", "collision"]
        predicted_failure_mode = "none"
        
        if "FAILURE_MODE:" in response_upper:
            mode_text = response.split("FAILURE_MODE:")[1].split("\n")[0].strip().lower()
            for mode in failure_modes:
                if mode in mode_text:
                    predicted_failure_mode = mode
                    break
        
        return predicted_outcome, predicted_failure_mode

def run_comprehensive_evaluation():
    """Run complete evaluation pipeline"""
    
    print("="*80)
    print("COMPREHENSIVE PHYSICS INTELLIGENCE EVALUATION")
    print("="*80)
    print()
    
    # Configuration
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = EvaluationConfig(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        test_failure_count=19,
        test_success_count=15,
        baseline_model=BASELINE_MODEL,
        lora_model=f"{LORA_BASE_MODEL} + LoRA",
        random_seed=42
    )
    
    # Load test data
    print("Loading test scenarios...")
    failure_scenarios = load_scenarios("failure_scenarios_dataset.json")
    random.seed(42)
    random.shuffle(failure_scenarios)
    test_failures = failure_scenarios[30:49]  # 19 failure scenarios
    
    with open("success_scenarios_dataset.json") as f:
        success_data = json.load(f)
        test_successes = [SuccessScenario(**s) for s in success_data]
    
    print(f"✅ Loaded {len(test_failures)} failure scenarios")
    print(f"✅ Loaded {len(test_successes)} success scenarios")
    print(f"Total test set: {len(test_failures) + len(test_successes)} scenarios")
    print()
    
    # Evaluate baseline
    print("="*80)
    print("EVALUATING BASELINE MODEL")
    print("="*80)
    baseline_evaluator = BaselineEvaluator(BASELINE_MODEL)
    baseline_results = []
    
    print("Testing on failure scenarios...")
    for i, scenario in enumerate(test_failures, 1):
        print(f"  {i}/{len(test_failures)}: {scenario.object_name}", end="\r")
        result = baseline_evaluator.predict(scenario, is_failure=True)
        baseline_results.append(result)
    
    print(f"\nTesting on success scenarios...")
    for i, scenario in enumerate(test_successes, 1):
        print(f"  {i}/{len(test_successes)}: {scenario.object_name}", end="\r")
        result = baseline_evaluator.predict(scenario, is_failure=False)
        baseline_results.append(result)
    
    print("\n")
    
    # Evaluate LoRA
    print("="*80)
    print("EVALUATING LORA MODEL")
    print("="*80)
    
    if HAS_TRANSFORMERS and LORA_ADAPTER_PATH.exists():
        lora_evaluator = LoRAEvaluator(LORA_BASE_MODEL, LORA_ADAPTER_PATH)
        lora_results = []
        
        print("Testing on failure scenarios...")
        for i, scenario in enumerate(test_failures, 1):
            print(f"  {i}/{len(test_failures)}: {scenario.object_name}", end="\r")
            result = lora_evaluator.predict(scenario, is_failure=True)
            lora_results.append(result)
        
        print(f"\nTesting on success scenarios...")
        for i, scenario in enumerate(test_successes, 1):
            print(f"  {i}/{len(test_successes)}: {scenario.object_name}", end="\r")
            result = lora_evaluator.predict(scenario, is_failure=False)
            lora_results.append(result)
        
        print("\n")
    else:
        print("⚠️  LoRA model not available, skipping")
        lora_results = None
    
    # Generate report
    generate_report(config, baseline_results, lora_results, test_failures, test_successes)
    
    print("✅ Evaluation complete!")
    print(f"📄 Results saved to {RESULTS_DIR}/")

def generate_report(config, baseline_results, lora_results, test_failures, test_successes):
    """Generate comprehensive evaluation report"""
    
    # Calculate metrics
    def calculate_metrics(results):
        total = len(results)
        binary_correct = sum(1 for r in results if r.is_correct_binary)
        
        # Failure mode accuracy (only for actual failures)
        failure_results = [r for r in results if r.actual_outcome == "failure"]
        failure_mode_correct = sum(1 for r in failure_results if r.is_correct_failure_mode)
        
        return {
            "binary_accuracy": (binary_correct / total * 100) if total > 0 else 0,
            "binary_correct": binary_correct,
            "failure_mode_accuracy": (failure_mode_correct / len(failure_results) * 100) if failure_results else 0,
            "failure_mode_correct": failure_mode_correct,
            "total": total,
            "total_failures": len(failure_results)
        }
    
    baseline_metrics = calculate_metrics(baseline_results)
    lora_metrics = calculate_metrics(lora_results) if lora_results else None
    
    # Save detailed results
    results_data = {
        "config": asdict(config),
        "baseline": {
            "metrics": baseline_metrics,
            "predictions": [asdict(r) for r in baseline_results]
        }
    }
    
    if lora_metrics:
        results_data["lora"] = {
            "metrics": lora_metrics,
            "predictions": [asdict(r) for r in lora_results]
        }
    
    results_file = RESULTS_DIR / f"evaluation_{config.run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Detailed results: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"Baseline Model: {config.baseline_model}")
    print(f"  Binary Accuracy:       {baseline_metrics['binary_accuracy']:.1f}% ({baseline_metrics['binary_correct']}/{baseline_metrics['total']})")
    print(f"  Failure Mode Accuracy: {baseline_metrics['failure_mode_accuracy']:.1f}% ({baseline_metrics['failure_mode_correct']}/{baseline_metrics['total_failures']})")
    print()
    
    if lora_metrics:
        print(f"LoRA Model: {config.lora_model}")
        print(f"  Binary Accuracy:       {lora_metrics['binary_accuracy']:.1f}% ({lora_metrics['binary_correct']}/{lora_metrics['total']})")
        print(f"  Failure Mode Accuracy: {lora_metrics['failure_mode_accuracy']:.1f}% ({lora_metrics['failure_mode_correct']}/{lora_metrics['total_failures']})")
        print()
        
        # Improvement
        binary_improvement = lora_metrics['binary_accuracy'] - baseline_metrics['binary_accuracy']
        mode_improvement = lora_metrics['failure_mode_accuracy'] - baseline_metrics['failure_mode_accuracy']
        print(f"Improvement:")
        print(f"  Binary:       {binary_improvement:+.1f}%")
        print(f"  Failure Mode: {mode_improvement:+.1f}%")

if __name__ == "__main__":
    run_comprehensive_evaluation()
