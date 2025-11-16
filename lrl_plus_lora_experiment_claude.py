"""
Scheduling Experiment with TIME-BASED OVERLAPS - CLAUDE HAIKU VERSION
======================================================================

Uses Claude 3.5 Haiku via Anthropic API instead of Ollama.
Much faster than 7B on CPU, and likely better performance.

CHANGES FROM 7B VERSION:
- Uses Claude 3.5 Haiku for Stages 1-3 (via API)
- Still uses Qwen2.5-7B-Instruct for Stage 4 LoRA (no Claude fine-tuning available)
- Expected runtime: 1-2 hours for complete experiment
- Requires ANTHROPIC_API_KEY environment variable
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Stage control flags - set to True to run that stage
RUN_STAGE_0_STUDENT_BASELINE = False   # New: Measure Qwen 7B before any training
RUN_STAGE_1_TEACHER_BASELINE = False  # Teacher (Claude) baseline
RUN_STAGE_2_BOOTSTRAP = True         # Teacher learns strategy
RUN_STAGE_3_LRL_TEST = True          # Teacher applies learned strategy
RUN_STAGE_4_LORA = True              # Student learns from teacher via LoRA

# Model paths - can be HuggingFace ID or local path
QWEN_MODEL_PATH = "./models/Qwen2.5-7B-Instruct"  # Local model (run download_model.py first)

# =============================================================================

import json
import time
import re
import random
import torch
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Silence transformers warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

# Anthropic API
from anthropic import Anthropic

# Add LoRA imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset

# Silence transformers logging after imports
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass
class SchedulingProblem:
    """Time-based scheduling: meetings with time ranges"""
    problem_id: str
    meetings: List[Tuple[int, int]]  # List of (start, end) times
    num_rooms: int
    is_solvable: bool
    max_simultaneous: int  # Peak overlap
    difficulty: str


class SchedulingSolver:
    """Solver with Linguistic Reinforcement Learning - Claude Haiku version"""
    
    def __init__(self, enable_learning: bool = False):
        # Initialize Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set!")
        
        self.client = Anthropic(api_key=api_key)
        self.model_name = "claude-3-5-haiku-20241022"
        self.enable_learning = enable_learning
        
        # Initial strategy
        self.strategy = """To determine if meetings can be scheduled:
1. Check which meetings overlap in time
2. Count the maximum number of meetings happening at the same time
3. If max_simultaneous <= num_rooms: YES (solvable)
4. If max_simultaneous > num_rooms: NO (impossible)

Key insight: A meeting ending at time T does NOT conflict with a meeting starting at time T."""
        
        # Learning state
        self.journal = ""
        self.problems_solved = 0
        self.batch_results = []
    
    def _call_llm(self, prompt: str, timeout: int = 20) -> str:
        """Call Claude API for inference"""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"NO - error: {str(e)}"
    
    def solve(self, problem: SchedulingProblem) -> Dict[str, Any]:
        """Solve a scheduling problem"""
        start_time = time.time()
        
        # Format meetings list
        meetings_str = ", ".join([f"[{s}-{e}]" for s, e in problem.meetings])
        
        prompt = f"""PROBLEM STRUCTURE (Ground Truth):
- Meetings have ONLY time ranges (integers, e.g., 9-11 means 9:00 to 11:00)
- Rooms are countable resources (no sizes, capacities, priorities)
- A meeting ending at time T does NOT conflict with a meeting starting at time T
- NO other features exist (no priorities, durations, preferences, etc.)

CURRENT STRATEGY:
{self.strategy}

SCHEDULING PROBLEM:
Meetings: {meetings_str}
Available rooms: {problem.num_rooms}

Can all meetings be scheduled without conflicts?

Answer with YES or NO, followed by brief explanation. Focus on:
1. Which meetings overlap
2. Maximum simultaneous meetings
3. Comparison to available rooms"""

        response = self._call_llm(prompt, timeout=25)
        elapsed = time.time() - start_time
        
        # Parse response
        answer = "YES" if "YES" in response.upper()[:50] else "NO"
        correct = (answer == "YES") == problem.is_solvable
        
        result = {
            "problem_id": problem.problem_id,
            "answer": answer,
            "correct": correct,
            "response": response,
            "elapsed": elapsed,
            "expected": "YES" if problem.is_solvable else "NO",
            "problem": f"{meetings_str} | {problem.num_rooms} rooms",
            "max_simultaneous": problem.max_simultaneous
        }
        
        # Log individual thought for review
        self._log_thought(problem, result)
        
        return result
    
    def _log_thought(self, problem: SchedulingProblem, result: Dict):
        """Log individual problem-solving thought"""
        log_entry = f"""
{'='*80}
Problem: {result['problem']} (max_simultaneous={problem.max_simultaneous})
Expected: {result['expected']} | Got: {result['answer']} | {'✓' if result['correct'] else '✗'}
Time: {result['elapsed']:.1f}s

AI Response:
{result['response']}

Current Strategy Being Used:
{self.strategy[:300]}...
{'='*80}
"""
        # Append to thoughts log
        with open("scheduling_thoughts_claude.log", "a") as f:
            f.write(log_entry)
    
    def write_journal_entry(self, batch_results: List[Dict], batch_num: int):
        """Write reflective journal about batch performance"""
        correct_count = sum(1 for r in batch_results if r["correct"])
        accuracy = correct_count / len(batch_results)
        
        # Prepare examples with details
        examples = []
        for r in batch_results[:5]:  # First 5 examples
            examples.append(
                f"- Problem: {r['problem']}\n"
                f"  Expected: {r['expected']}, Got: {r['answer']}, "
                f"Correct: {r['correct']}, Max_sim: {r.get('max_simultaneous', '?')}"
            )
        
        prompt = f"""PROBLEM STRUCTURE (Ground Truth):
- Meetings have ONLY time ranges (integers)
- Rooms are countable
- Meeting ending at T does NOT conflict with meeting starting at T
- NO priorities, sizes, durations, or other features

BATCH {batch_num} PERFORMANCE:
Accuracy: {correct_count}/{len(batch_results)} = {accuracy:.1%}

Examples:
{chr(10).join(examples)}

Write a journal entry analyzing:
1. What patterns led to correct answers?
2. What mistakes happened and why?
3. What edge cases need attention?
4. What insights can improve the strategy?

Focus ONLY on actual problem features (time ranges, room counting).
DO NOT hallucinate non-existent attributes."""

        journal_entry = self._call_llm(prompt, timeout=30)
        
        # Create structured log entry
        log_entry = f"""
{'#'*80}
BATCH {batch_num} JOURNAL ENTRY
Accuracy: {correct_count}/{len(batch_results)} = {accuracy:.1%}
{'#'*80}

EXAMPLES ANALYZED:
{chr(10).join(examples)}

AI REFLECTION:
{journal_entry}

{'#'*80}
"""
        
        # Append to both internal journal and log file
        self.journal += f"\n\n=== BATCH {batch_num} ===\n{journal_entry}"
        
        with open("scheduling_journal_claude.log", "a") as f:
            f.write(log_entry)
        
        print(f"   📝 Journal entry written (logged to scheduling_journal_claude.log)")
    
    def _distill_strategy(self):
        """Extract improved strategy from journal"""
        prompt = f"""Based on these journal entries:
{self.journal[-5000:]}  

Extract the most effective scheduling strategies.

Focus on:
- How to count simultaneous meetings correctly
- How to detect time overlaps
- Edge cases (boundaries, single meetings, etc.)
- Solvability determination rules

Provide concise, actionable strategies. Stay grounded in actual problem constraints:
- Meetings have time ranges only
- Rooms are countable only
- No other features exist"""

        new_strategy = self._call_llm(prompt, timeout=30)
        
        # Log strategy evolution
        log_entry = f"""
{'*'*80}
STRATEGY DISTILLATION (from journal analysis)
{'*'*80}

OLD STRATEGY:
{self.strategy}

NEW STRATEGY:
{new_strategy}

{'*'*80}
"""
        
        with open("scheduling_strategy_evolution_claude.log", "a") as f:
            f.write(log_entry)
        
        self.strategy = new_strategy
        print(f"   🧠 Strategy updated (logged to scheduling_strategy_evolution_claude.log)")
    
    def learn_from_batch(self, batch_results: List[Dict], batch_num: int):
        """Learn from batch if learning enabled"""
        if not self.enable_learning:
            return
        
        self.batch_results = batch_results
        self.write_journal_entry(batch_results, batch_num)
        
        # Distill every 5 batches
        if batch_num % 5 == 0:
            self._distill_strategy()


class LoRASolver:
    """Solver using LoRA fine-tuned adapter - still uses Qwen 7B"""
    
    def __init__(self, adapter_path: str, base_model: str = None):
        if base_model is None:
            base_model = QWEN_MODEL_PATH
        
        print(f"📥 Loading LoRA adapter from {adapter_path}...")
        print(f"   Base model: {base_model}")
        
        # Check if GPU is available
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16  # FP16 for GPU
            print(f"   ✅ GPU detected - using CUDA for fast inference")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32  # Full precision for CPU
            print(f"   ⚠️  No GPU detected - running on CPU (will be SLOW)")
        
        # Check if it's a local path or HF model
        use_local = base_model.startswith("./") or base_model.startswith("/")
        if use_local:
            print(f"   📂 Loading base model from local path...")
        else:
            print(f"   📥 Downloading base model if not cached (~14GB, one-time)...")
        
        # Load base model (GPU if available, CPU otherwise)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=use_local
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        print("✅ LoRA model loaded")
    
    def solve(self, problem: SchedulingProblem) -> Dict[str, Any]:
        """Solve using LoRA model"""
        meetings_str = ", ".join([f"[{s}-{e}]" for s, e in problem.meetings])
        
        # Use same strategy format as training
        prompt = f"""Strategy:
To determine if meetings can be scheduled:
1. Check which meetings overlap in time
2. Count the maximum number of meetings happening at the same time
3. If max_simultaneous <= num_rooms: YES (solvable)
4. If max_simultaneous > num_rooms: NO (impossible)

Key: A meeting ending at time T does NOT conflict with one starting at T.

Problem:
Meetings: {meetings_str}
Available rooms: {problem.num_rooms}

Can all meetings be scheduled?

Answer:"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        answer = "YES" if "YES" in response.upper()[:20] else "NO"
        correct = (answer == "YES") == problem.is_solvable
        
        return {
            "problem_id": problem.problem_id,
            "answer": answer,
            "correct": correct,
            "expected": "YES" if problem.is_solvable else "NO",
            "response": response
        }


class QwenBaselineSolver:
    """Baseline solver using raw Qwen 7B without any training"""
    
    def __init__(self, base_model: str = None):
        if base_model is None:
            base_model = QWEN_MODEL_PATH
        
        print(f"📥 Loading Qwen baseline model...")
        print(f"   Model: {base_model}")
        
        # Check if GPU is available
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16  # FP16 for GPU
            print(f"   ✅ GPU detected - using CUDA for fast inference")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32  # Full precision for CPU
            print(f"   ⚠️  No GPU detected - running on CPU (will be SLOW)")
        
        # Check if it's a local path or HF model
        use_local = base_model.startswith("./") or base_model.startswith("/")
        if use_local:
            print(f"   📂 Loading from local path...")
        else:
            print(f"   📥 Downloading from HuggingFace if not cached (~14GB, one-time)...")
        
        # Load base model (GPU if available, CPU otherwise)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=use_local
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            trust_remote_code=True,
            local_files_only=use_local
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Qwen baseline model loaded")
    
    def solve(self, problem: SchedulingProblem) -> Dict[str, Any]:
        """Solve using raw Qwen model (no strategy, no training)"""
        meetings_str = ", ".join([f"[{s}-{e}]" for s, e in problem.meetings])
        
        # Simple prompt without any strategy guidance
        prompt = f"""Problem:
Meetings: {meetings_str}
Available rooms: {problem.num_rooms}

Can all meetings be scheduled without conflicts?

Answer with YES or NO:"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        answer = "YES" if "YES" in response.upper()[:20] else "NO"
        correct = (answer == "YES") == problem.is_solvable
        
        return {
            "problem_id": problem.problem_id,
            "answer": answer,
            "correct": correct,
            "expected": "YES" if problem.is_solvable else "NO",
            "response": response
        }


def train_lora_adapter(strategy: str, num_examples: int = 100, output_dir: str = "./lora_adapter_claude") -> str:
    """Train LoRA adapter on synthetic examples - still uses Qwen 7B"""
    print("\n" + "="*80)
    print("🔧 TRAINING LoRA ADAPTER")
    print("="*80)
    
    # Generate training data
    print(f"Generating {num_examples} training examples...")
    training_data = []
    random.seed(999)  # Use different seed to avoid test overlap
    
    for i in range(num_examples):
        difficulty = random.choice(["EASY", "MEDIUM", "HARD"])
        
        if difficulty == "EASY":
            num_meetings = 2
            num_rooms = random.choice([2, 3])
        elif difficulty == "MEDIUM":
            num_meetings = random.choice([3, 4])
            num_rooms = random.choice([2, 3])
        else:
            num_meetings = random.choice([4, 5, 6])
            num_rooms = random.choice([2, 3, 4])
        
        meetings = []
        for _ in range(num_meetings):
            start = random.randint(9, 15)
            duration = random.randint(1, 3)
            end = start + duration
            meetings.append((start, end))
        
        max_sim = _calculate_max_simultaneous(meetings)
        is_solvable = max_sim <= num_rooms
        answer = "YES" if is_solvable else "NO"
        meetings_str = ", ".join([f"[{s}-{e}]" for s, e in meetings])
        
        prompt = f"""Strategy:
{strategy}

Problem:
Meetings: {meetings_str}
Available rooms: {num_rooms}

Can all meetings be scheduled?

Answer:"""
        
        training_data.append({"text": prompt + f" {answer}"})
    
    print(f"✅ Generated {len(training_data)} examples")
    
    # Load base model for CPU
    print(f"Loading base model: {QWEN_MODEL_PATH}")
    print("⚠️  Training will be SLOW on CPU - expect 2-4 hours for LoRA training")
    
    # Check if it's a local path or HF model
    use_local = QWEN_MODEL_PATH.startswith("./") or QWEN_MODEL_PATH.startswith("/")
    if use_local:
        print(f"📂 Loading from local path...")
    else:
        print(f"📥 Downloading from HuggingFace if not cached (~14GB, one-time)...")
    
    # No quantization for CPU - load full precision
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_PATH,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Full precision
        low_cpu_mem_usage=True,
        local_files_only=use_local
    )
    # Enable gradient computation for training (replaces prepare_model_for_kbit_training for non-quantized models)
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_PATH, 
        trust_remote_code=True,
        local_files_only=use_local
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    print(f"✅ LoRA adapter configured")
    
    # Prepare dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Training arguments - FORCE CPU ONLY
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",  # Changed from paged_adamw_8bit for CPU
        gradient_checkpointing=True,
        report_to="none",
        no_cuda=True,  # FORCE CPU - do not use GPU
        use_cpu=True   # Explicitly use CPU
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("🚀 Starting LoRA training...")
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ LoRA adapter saved to {output_dir}")
    return output_dir


def generate_problems(num_problems: int = 100, seed: int = 42) -> List[SchedulingProblem]:
    """Generate time-based scheduling problems"""
    random.seed(seed)
    
    problems = []
    problem_id = 0
    
    # Distribution: 33% EASY, 33% MEDIUM, 33% HARD
    difficulties = ["EASY"] * (num_problems // 3) + \
                   ["MEDIUM"] * (num_problems // 3) + \
                   ["HARD"] * (num_problems - 2 * (num_problems // 3))
    random.shuffle(difficulties)
    
    for diff in difficulties:
        if diff == "EASY":
            num_meetings = 2
            num_rooms = random.choice([2, 3])
        elif diff == "MEDIUM":
            num_meetings = random.choice([3, 4])
            num_rooms = random.choice([2, 3])
        else:  # HARD
            num_meetings = random.choice([4, 5, 6])
            num_rooms = random.choice([2, 3, 4])
        
        # Generate meetings with time ranges
        meetings = []
        for _ in range(num_meetings):
            start = random.randint(9, 15)  # 9am to 3pm starts
            duration = random.randint(1, 3)  # 1-3 hours
            end = start + duration
            meetings.append((start, end))
        
        # Calculate max simultaneous meetings
        max_sim = _calculate_max_simultaneous(meetings)
        is_solvable = max_sim <= num_rooms
        
        problems.append(SchedulingProblem(
            problem_id=f"{diff.lower()}_{problem_id}",
            meetings=meetings,
            num_rooms=num_rooms,
            is_solvable=is_solvable,
            max_simultaneous=max_sim,
            difficulty=diff
        ))
        problem_id += 1
    
    return problems


def _calculate_max_simultaneous(meetings: List[Tuple[int, int]]) -> int:
    """Calculate maximum number of meetings happening at the same time"""
    if not meetings:
        return 0
    
    # Create events (start and end)
    events = []
    for start, end in meetings:
        events.append((start, 1))  # Meeting starts
        events.append((end, -1))   # Meeting ends
    
    # Sort events (at same time, process ends before starts)
    events.sort(key=lambda x: (x[0], x[1]))
    
    max_simultaneous = 0
    current = 0
    
    for time, delta in events:
        current += delta
        max_simultaneous = max(max_simultaneous, current)
    
    return max_simultaneous


def run_stage(stage_name: str, solver, problems: List[SchedulingProblem],
              batch_size: int = 10) -> Dict:
    """Run one stage of the experiment"""
    print()
    print("="*80)
    print(f"🧪 {stage_name}")
    print("="*80)
    
    # Detect solver type and show appropriate info
    if hasattr(solver, 'enable_learning'):
        print(f"Model: Claude 3.5 Haiku | Learning: {solver.enable_learning}")
    elif isinstance(solver, QwenBaselineSolver):
        print(f"Model: Qwen2.5-7B-Instruct (Baseline - no training)")
    elif isinstance(solver, LoRASolver):
        print(f"Model: Qwen2.5-7B-Instruct (LoRA fine-tuned)")
    else:
        print(f"Model: Unknown")
    
    print(f"Problems: {len(problems)} total")
    print("="*80)
    print()
    
    results = []
    num_batches = (len(problems) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(problems))
        batch = problems[start_idx:end_idx]
        
        print(f"📦 Batch {batch_idx + 1}/{num_batches}...", end=" ", flush=True)
        
        batch_results = []
        for problem in batch:
            result = solver.solve(problem)
            batch_results.append(result)
            results.append(result)
        
        # Calculate batch accuracy
        batch_correct = sum(1 for r in batch_results if r["correct"])
        batch_acc = batch_correct / len(batch_results)
        print(f"{batch_correct}/{len(batch_results)} correct ({batch_acc:.1%})")
        
        # Learn from batch (only if solver supports it)
        if hasattr(solver, 'learn_from_batch'):
            solver.learn_from_batch(batch_results, batch_idx + 1)
    
    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    
    # By difficulty
    by_difficulty = {}
    for r in results:
        diff = next((p.difficulty for p in problems if p.problem_id == r["problem_id"]), "UNKNOWN")
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}
        by_difficulty[diff]["total"] += 1
        if r["correct"]:
            by_difficulty[diff]["correct"] += 1
    
    print()
    print("="*80)
    print(f"📊 {stage_name} RESULTS")
    print("="*80)
    print(f"Overall: {correct}/{total} = {accuracy:.1%}")
    print()
    print("By Difficulty:")
    for diff in ["EASY", "MEDIUM", "HARD"]:
        if diff in by_difficulty:
            d = by_difficulty[diff]
            d_acc = d["correct"] / d["total"]
            print(f"  {diff:6s}: {d['correct']}/{d['total']} = {d_acc:.1%}")
    print("="*80)
    print()
    
    return {
        "stage": stage_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_difficulty": by_difficulty,
        "results": results
    }


def main():
    """Run the five-stage experiment with Claude and Qwen"""
    print("\n" + "="*80)
    print("LINGUISTIC RL FOR SCHEDULING - CLAUDE 3.5 HAIKU VERSION")
    print("="*80)
    print()
    
    # Show enabled stages
    print("📋 Enabled Stages:")
    if RUN_STAGE_0_STUDENT_BASELINE:
        print("   ✅ Stage 0: Student Baseline (Qwen 7B - no training)")
    if RUN_STAGE_1_TEACHER_BASELINE:
        print("   ✅ Stage 1: Teacher Baseline (Claude Haiku - no LRL)")
    if RUN_STAGE_2_BOOTSTRAP:
        print("   ✅ Stage 2: Bootstrap (Claude learns via LRL)")
    if RUN_STAGE_3_LRL_TEST:
        print("   ✅ Stage 3: LRL Test (Claude uses learned strategy)")
    if RUN_STAGE_4_LORA:
        print("   ✅ Stage 4: LoRA (Qwen trained on Claude's strategy)")
    print()
    
    # Check API key if needed
    if RUN_STAGE_1_TEACHER_BASELINE or RUN_STAGE_2_BOOTSTRAP or RUN_STAGE_3_LRL_TEST:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set!")
            print()
            print("Set it with:")
            print("  export ANTHROPIC_API_KEY='your-api-key-here'")
            print()
            return
        print("✅ ANTHROPIC_API_KEY found")
        print()
    
    # Clear previous logs
    for log_file in ["scheduling_thoughts_claude.log", "scheduling_journal_claude.log", "scheduling_strategy_evolution_claude.log"]:
        Path(log_file).unlink(missing_ok=True)
    
    print("📝 Logging enabled:")
    print("   - scheduling_thoughts_claude.log (every problem's reasoning)")
    print("   - scheduling_journal_claude.log (batch reflections)")
    print("   - scheduling_strategy_evolution_claude.log (strategy changes)")
    print()
    
    # Generate problems
    print("📋 Generating problems...")
    train_problems = generate_problems(num_problems=100, seed=42)
    test_problems = generate_problems(num_problems=150, seed=123)
    print(f"   Training: {len(train_problems)} problems")
    print(f"   Test: {len(test_problems)} problems")
    print()
    
    all_results = {}
    learned_strategy = None
    
    # STAGE 0: Student Baseline (Qwen 7B - no training)
    if RUN_STAGE_0_STUDENT_BASELINE:
        print("🚀 Starting STAGE 0: Student Baseline")
        print("   Qwen 7B zero-shot performance (no training)")
        qwen_baseline_solver = QwenBaselineSolver()
        stage0_results = run_stage("Stage 0: Student Baseline (Qwen 7B)", qwen_baseline_solver, test_problems)
        all_results["student_baseline"] = stage0_results
        
        time.sleep(2)
    
    # STAGE 1: Teacher Baseline (Claude - No Learning)
    if RUN_STAGE_1_TEACHER_BASELINE:
        print("🚀 Starting STAGE 1: Teacher Baseline (No LRL)")
        print("   Claude Haiku zero-shot performance on test set")
        baseline_solver = SchedulingSolver(enable_learning=False)
        stage1_results = run_stage("Stage 1: Teacher Baseline (Claude Haiku - No LRL)", baseline_solver, test_problems)
        all_results["teacher_baseline"] = stage1_results
        
        time.sleep(2)
    
    # STAGE 2: Bootstrap (Learning)
    if RUN_STAGE_2_BOOTSTRAP:
        print("🚀 Starting STAGE 2: Bootstrap Learning")
        print("   Claude learning strategy from training set")
        bootstrap_solver = SchedulingSolver(enable_learning=True)
        stage2_results = run_stage("Stage 2: Bootstrap", bootstrap_solver, train_problems)
        all_results["bootstrap"] = stage2_results
        
        # Save learned strategy
        learned_strategy = bootstrap_solver.strategy
        strategy_file = Path("scheduling_lrl_strategy_claude.txt")
        with open(strategy_file, 'w') as f:
            f.write(learned_strategy)
        print(f"💾 Learned strategy saved to: {strategy_file}")
        print()
        
        # Save journal
        journal_file = Path("scheduling_lrl_journal_claude.txt")
        with open(journal_file, 'w') as f:
            f.write(f"=== BOOTSTRAP JOURNAL ===\n{bootstrap_solver.journal}")
        print(f"💾 Learning journal saved to: {journal_file}")
        
        time.sleep(2)
    
    # Load strategy if Stage 2 was skipped
    if not RUN_STAGE_2_BOOTSTRAP and (RUN_STAGE_3_LRL_TEST or RUN_STAGE_4_LORA):
        strategy_file = Path("scheduling_lrl_strategy_claude.txt")
        if strategy_file.exists():
            with open(strategy_file, 'r') as f:
                learned_strategy = f.read()
            print(f"📥 Loaded existing strategy from: {strategy_file}")
        else:
            print("⚠️  Warning: No saved strategy found. Using default strategy.")
            learned_strategy = """To determine if meetings can be scheduled:
1. Check which meetings overlap in time
2. Count the maximum number of meetings happening at the same time
3. If max_simultaneous <= num_rooms: YES (solvable)
4. If max_simultaneous > num_rooms: NO (impossible)

Key insight: A meeting ending at time T does NOT conflict with a meeting starting at time T."""
    
    # STAGE 3: Test with LRL
    if RUN_STAGE_3_LRL_TEST:
        print("🚀 Starting STAGE 3: Test with LRL")
        print("   Claude using learned strategy on test set")
        lrl_solver = SchedulingSolver(enable_learning=False)
        lrl_solver.strategy = learned_strategy  # Use learned strategy
        stage3_results = run_stage("Stage 3: Test with LRL", lrl_solver, test_problems)
        all_results["lrl"] = stage3_results
        
        time.sleep(2)
    
    # STAGE 4: LoRA Fine-tuning
    if RUN_STAGE_4_LORA:
        print("🚀 Starting STAGE 4: LoRA Fine-tuning")
        print("   Training LoRA adapter on learned strategy")
        print("   ⚠️  Using Qwen 7B model on CPU - this will take 2-4 hours!")
        adapter_path = train_lora_adapter(learned_strategy, num_examples=100, output_dir="./lora_adapter_claude")
        time.sleep(2)
        
        # Test LoRA on same test set
        print("🚀 Testing LoRA adapter on test set")
        lora_solver = LoRASolver(adapter_path=adapter_path)
        
        # Run LoRA stage
        print()
        print("="*80)
        print("🧪 Stage 4: LoRA Test")
        print("="*80)
        print(f"Model: LoRA-tuned Qwen2.5-7B-Instruct (CPU)")
        print(f"Problems: {len(test_problems)} total")
        print("="*80)
        print()
        
        lora_results = []
        batch_size = 10
        num_batches = (len(test_problems) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_problems))
            batch = test_problems[start_idx:end_idx]
            
            print(f"📦 Batch {batch_idx + 1}/{num_batches}...", end=" ", flush=True)
            
            batch_results = []
            for problem in batch:
                result = lora_solver.solve(problem)
                batch_results.append(result)
                lora_results.append(result)
            
            batch_correct = sum(1 for r in batch_results if r["correct"])
            batch_acc = batch_correct / len(batch_results)
            print(f"{batch_correct}/{len(batch_results)} correct ({batch_acc:.1%})")
        
        # Calculate LoRA statistics
        total = len(lora_results)
        correct = sum(1 for r in lora_results if r["correct"])
        lora_accuracy = correct / total if total > 0 else 0
        
        by_difficulty = {}
        for r in lora_results:
            diff = next((p.difficulty for p in test_problems if p.problem_id == r["problem_id"]), "UNKNOWN")
            if diff not in by_difficulty:
                by_difficulty[diff] = {"correct": 0, "total": 0}
            by_difficulty[diff]["total"] += 1
            if r["correct"]:
                by_difficulty[diff]["correct"] += 1
        
        print()
        print("="*80)
        print("📊 Stage 4: LoRA Test RESULTS")
        print("="*80)
        print(f"Overall: {correct}/{total} = {lora_accuracy:.1%}")
        print()
        print("By Difficulty:")
        for diff in ["EASY", "MEDIUM", "HARD"]:
            if diff in by_difficulty:
                d = by_difficulty[diff]
                d_acc = d["correct"] / d["total"]
                print(f"  {diff:6s}: {d['correct']}/{d['total']} = {d_acc:.1%}")
        print("="*80)
        print()
        
        stage4_results = {
            "stage": "Stage 4: LoRA Test",
            "total": total,
            "correct": correct,
            "accuracy": lora_accuracy,
            "by_difficulty": by_difficulty,
            "results": lora_results
        }
        all_results["lora"] = stage4_results
    
    # Final comparison
    print()
    print("="*80)
    print("📊 FINAL COMPARISON - CLAUDE 3.5 HAIKU + QWEN 7B")
    print("="*80)
    print()
    
    # Show only stages that ran
    if RUN_STAGE_0_STUDENT_BASELINE and "student_baseline" in all_results:
        print(f"Stage 0 - Student Baseline:   {all_results['student_baseline']['accuracy']:.1%}")
    if RUN_STAGE_1_TEACHER_BASELINE and "teacher_baseline" in all_results:
        print(f"Stage 1 - Teacher Baseline:   {all_results['teacher_baseline']['accuracy']:.1%}")
    if RUN_STAGE_2_BOOTSTRAP and "bootstrap" in all_results:
        print(f"Stage 2 - Bootstrap:          {all_results['bootstrap']['accuracy']:.1%}")
    if RUN_STAGE_3_LRL_TEST and "lrl" in all_results:
        print(f"Stage 3 - Test with LRL:      {all_results['lrl']['accuracy']:.1%}")
    if RUN_STAGE_4_LORA and "lora" in all_results:
        print(f"Stage 4 - Test with LoRA:     {all_results['lora']['accuracy']:.1%}")
    
    print()
    print("Knowledge Transfer Analysis:")
    if "student_baseline" in all_results and "lora" in all_results:
        improvement = all_results["lora"]["accuracy"] - all_results["student_baseline"]["accuracy"]
        print(f"  Student improvement (LoRA vs Baseline): {improvement:+.1%}")
    if "teacher_baseline" in all_results and "lora" in all_results:
        comparison = all_results["lora"]["accuracy"] - all_results["teacher_baseline"]["accuracy"]
        print(f"  Student vs Teacher: {comparison:+.1%}")
    if "teacher_baseline" in all_results and "lrl" in all_results:
        lrl_impact = all_results["lrl"]["accuracy"] - all_results["teacher_baseline"]["accuracy"]
        print(f"  LRL impact on Teacher: {lrl_impact:+.1%}")
    
    print()
    print("="*80)
    print()
    print("📝 Review logs:")
    print("   - scheduling_thoughts_claude.log (all problem reasoning)")
    print("   - scheduling_journal_claude.log (batch reflections)")
    print("   - scheduling_strategy_evolution_claude.log (strategy changes)")
    print("="*80)
    
    # Save results
    results_file = Path("scheduling_lrl_plus_lora_results_claude.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
