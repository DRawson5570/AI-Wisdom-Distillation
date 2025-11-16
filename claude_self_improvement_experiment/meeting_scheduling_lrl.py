#!/usr/bin/env python3
"""
Meeting Scheduling Knowledge Transfer Experiment
=================================================

Re-running the original breakthrough experiment with proper methodology:
- Teacher: Claude 3.5 Haiku (or same model for self-improvement)
- Student: Qwen 7B
- Domain: Time-based meeting scheduling with room constraints
- Target: Reproduce or validate the claimed 12% → 86.7% improvement

Uses the centralized framework to ensure validity.
"""

import os
import sys
import random
from typing import List, Tuple
from dataclasses import dataclass

# Add parent directory to path to import framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from framework import (
    Problem,
    TransferDomain,
    run_knowledge_transfer
)

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

CONFIG = {
    # Models
    'teacher_model': 'claude-3-5-haiku-20241022',  # Use Claude as teacher
    'student_model': 'Qwen/Qwen2.5-7B-Instruct',   # HF identifier for LoRA
    'student_model_ollama': 'qwen2.5:7b',          # Explicit Ollama tag for baseline
    
    # Alternative: Self-improvement with same model
    # 'teacher_model': 'Qwen/Qwen2.5-7B-Instruct',
    # 'student_model': 'Qwen/Qwen2.5-7B-Instruct',
    
    # Backend selection
    'use_ollama': False,  # Use Ollama for non-Claude models
    
    # Dataset sizes
    'num_train': 250,  # Training problems for LRL + LoRA
    'num_test': 150,   # Test set size
    'epochs': 3,       # LoRA training epochs
    
    # Stages to run
    'run_student_baseline': True,   # Stage 0: Student baseline
    'run_teacher_baseline': True,   # Stage 1: Teacher baseline
    'run_teacher_learning': True,   # Stage 2: Teacher LRL
    'run_student_transfer': True,   # Stage 3: LoRA transfer
    'run_prompt_injection': True,   # Stage 4: Prompt only
    
    # Output
    'results_dir': './results_meeting_scheduling',
    'verbose': True,
}


# ============================================================================
# PROBLEM DEFINITION
# ============================================================================

@dataclass
class MeetingSchedulingProblem(Problem):
    """
    Time-based scheduling problem:
    - N meetings with time ranges [start, end]
    - M rooms available
    - Question: Can all meetings be scheduled without conflicts?
    
    Key rule: A meeting ending at time T does NOT conflict with
    a meeting starting at time T (room becomes available immediately)
    """
    meetings: List[Tuple[int, int]]  # List of (start, end) times
    num_rooms: int
    is_solvable: bool
    max_simultaneous: int  # Peak overlap (ground truth)
    answer: str  # "YES" or "NO"
    
    def format_for_prompt(self) -> str:
        meetings_str = ", ".join([f"[{s}-{e}]" for s, e in self.meetings])
        return f"""You have {len(self.meetings)} meetings and {self.num_rooms} rooms.
Meeting times: {meetings_str}

IMPORTANT RULE: A meeting ending at time T does NOT conflict with a meeting starting at time T. The room becomes available immediately.

Question: Can all meetings be scheduled in the {self.num_rooms} available rooms without conflicts?

To solve: Find the maximum number of meetings happening at the same time. If this number is ≤ {self.num_rooms}, answer YES. Otherwise answer NO.

Answer with ONLY one word: YES or NO

Do NOT provide explanations, room assignments, or any other text. Just the word YES or NO."""


# ============================================================================
# DOMAIN IMPLEMENTATION
# ============================================================================

class MeetingSchedulingDomain(TransferDomain):
    """Time-based meeting scheduling domain"""
    
    def __init__(self):
        super().__init__("Meeting Scheduling")
    
    def generate_problems(self, num_problems: int, seed: int = 42) -> List[MeetingSchedulingProblem]:
        """Generate diverse scheduling problems"""
        
        random.seed(seed)
        problems = []
        
        # Mix of difficulties
        difficulties = ['EASY'] * (num_problems // 3) + \
                      ['MEDIUM'] * (num_problems // 3) + \
                      ['HARD'] * (num_problems - 2*(num_problems//3))
        random.shuffle(difficulties)
        
        for i, diff in enumerate(difficulties):
            if diff == 'EASY':
                # 3-4 meetings, 2-3 rooms, obvious answer
                num_meetings = random.randint(3, 4)
                num_rooms = random.randint(2, 3)
                time_range = 20
            elif diff == 'MEDIUM':
                # 5-7 meetings, 2-4 rooms, requires calculation
                num_meetings = random.randint(5, 7)
                num_rooms = random.randint(2, 4)
                time_range = 30
            else:  # HARD
                # 8-12 meetings, 3-5 rooms, complex overlaps
                num_meetings = random.randint(8, 12)
                num_rooms = random.randint(3, 5)
                time_range = 40
            
            # Generate meetings
            meetings = []
            for _ in range(num_meetings):
                start = random.randint(0, time_range - 5)
                duration = random.randint(2, 8)
                end = min(start + duration, time_range)
                meetings.append((start, end))
            
            # Calculate max simultaneous meetings
            max_sim = self._calculate_max_simultaneous(meetings)
            
            # Determine if solvable
            is_solvable = max_sim <= num_rooms
            answer = "YES" if is_solvable else "NO"
            
            problem = MeetingSchedulingProblem(
                problem_id=f"{diff.lower()}_{i}",
                description="",  # Will be set by format_for_prompt()
                difficulty=diff,
                metadata={"time_range": time_range},
                meetings=meetings,
                num_rooms=num_rooms,
                is_solvable=is_solvable,
                max_simultaneous=max_sim,
                answer=answer
            )
            # Set the full formatted description
            problem.description = problem.format_for_prompt()
            problems.append(problem)
        
        return problems
    
    def _calculate_max_simultaneous(self, meetings: List[Tuple[int, int]]) -> int:
        """Calculate maximum number of meetings happening at the same time"""
        events = []
        for start, end in meetings:
            events.append((start, 1))    # Meeting starts
            events.append((end, -1))     # Meeting ends
        
        events.sort()
        max_sim = 0
        current = 0
        
        for time, delta in events:
            current += delta
            max_sim = max(max_sim, current)
        
        return max_sim
    
    def evaluate_response(self, problem: MeetingSchedulingProblem, response: str) -> Tuple[bool, str]:
        """Check if response contains correct answer"""
        
        response_upper = response.upper()
        
        # Look for explicit YES or NO answers
        has_yes = "YES" in response_upper
        has_no = "NO" in response_upper
        
        # Enhanced parsing: look for common verbose patterns from Claude
        # Check for patterns like "The answer is YES" or "Answer: NO"
        import re
        
        # Try to find explicit answer statements (case-insensitive)
        answer_patterns = [
            r'\bANSWER[:\s]+YES\b',
            r'\bANSWER[:\s]+NO\b',
            r'\bRESULT[:\s]+YES\b',
            r'\bRESULT[:\s]+NO\b',
            r'\b(?:THE\s+)?ANSWER\s+IS[:\s]+YES\b',
            r'\b(?:THE\s+)?ANSWER\s+IS[:\s]+NO\b',
            r'^\s*YES\s*$',  # Just "YES" alone
            r'^\s*NO\s*$',   # Just "NO" alone
        ]
        
        extracted_answer = None
        for pattern in answer_patterns:
            match = re.search(pattern, response_upper)
            if match:
                if 'YES' in match.group():
                    extracted_answer = "YES"
                elif 'NO' in match.group():
                    extracted_answer = "NO"
                break
        
        # If we found an explicit answer, use it
        if extracted_answer:
            answer = extracted_answer
        # Otherwise fall back to presence detection
        elif has_yes and not has_no:
            answer = "YES"
        elif has_no and not has_yes:
            answer = "NO"
        elif has_yes and has_no:
            # Both present, take first one
            yes_pos = response_upper.find("YES")
            no_pos = response_upper.find("NO")
            answer = "YES" if yes_pos < no_pos else "NO"
        else:
            # Neither found - this is ambiguous, but we can't guess
            # Instead of defaulting, mark it as unclear
            answer = "UNCLEAR"
        
        is_correct = answer == problem.answer
        
        if is_correct:
            explanation = f"Correct: {problem.answer}"
        else:
            explanation = f"Expected: {problem.answer}, Got: {answer} (from: {response[:100]}...)"
        
        return is_correct, explanation
    
    def get_ground_truth(self, problem: MeetingSchedulingProblem) -> str:
        """Return the reference answer - just YES or NO"""
        return problem.answer


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Run meeting scheduling knowledge transfer experiment"""
    
    print("="*80)
    print("MEETING SCHEDULING KNOWLEDGE TRANSFER EXPERIMENT")
    print("="*80)
    print()
    print("Reproducing/validating the claimed breakthrough:")
    print("  Original claim: 12% → 86.7% improvement")
    print("  Method: LRL + LoRA transfer")
    print()
    
    # Create domain
    domain = MeetingSchedulingDomain()
    
    print("Configuration:")
    print(f"  Teacher: {CONFIG['teacher_model']}")
    print(f"  Student (HF): {CONFIG['student_model']}")
    print(f"  Student (Ollama): {CONFIG['student_model_ollama']}")
    print(f"  Training: {CONFIG['num_train']} problems")
    print(f"  Testing: {CONFIG['num_test']} problems")
    print(f"  Backend: {'Ollama' if CONFIG['use_ollama'] else 'HuggingFace'}")
    print()
    print("="*80)
    print()
    
    # Get API key if using Claude
    api_key = None
    if 'claude' in CONFIG['teacher_model'].lower():
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("⚠️  Warning: ANTHROPIC_API_KEY not set!")
            print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
            print()
    
    # Run experiment
    results = run_knowledge_transfer(
        domain=domain,
        teacher_model=CONFIG['teacher_model'],
        student_model=CONFIG['student_model'],
        student_model_ollama=CONFIG['student_model_ollama'],
        anthropic_api_key=api_key,
        use_ollama=CONFIG['use_ollama'],
        num_training=CONFIG['num_train'],
        num_test=CONFIG['num_test'],
        num_epochs=CONFIG['epochs'],
        run_student_baseline=CONFIG['run_student_baseline'],
        run_teacher_baseline=CONFIG['run_teacher_baseline'],
        run_teacher_learning=CONFIG['run_teacher_learning'],
        run_student_transfer=CONFIG['run_student_transfer'],
        run_prompt_injection=CONFIG['run_prompt_injection'],
        output_dir=CONFIG['results_dir'],
        verbose_file=CONFIG['verbose'],      # Write detailed Q&A to log file
        verbose_terminal=False                # Don't print Q&A to terminal (quieter)
    )
    
    # Display comprehensive results
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON - Meeting Scheduling Results")
    print("="*80)
    print()
    
    if 'student_baseline' in results:
        baseline = results['student_baseline']['accuracy'] * 100
        print(f"Stage 0 - Student Baseline:   {baseline:.1f}%")
    
    if 'teacher_baseline' in results:
        teacher = results['teacher_baseline']['accuracy'] * 100
        print(f"Stage 1 - Teacher Baseline:   {teacher:.1f}%")
    
    if 'teacher_learning' in results:
        teacher_lrl = results['teacher_learning']['accuracy'] * 100
        print(f"Stage 2 - Teacher Learning:   {teacher_lrl:.1f}%")
    
    if 'student_transfer' in results:
        lora = results['student_transfer']['accuracy'] * 100
        print(f"Stage 3 - LoRA Transfer:      {lora:.1f}%")
        
        if 'student_baseline' in results:
            improvement = lora - baseline
            print()
            print(f"💡 Improvement: +{improvement:.1f}% ({baseline:.1f}% → {lora:.1f}%)")
            print()
            print("Comparison to original claim:")
            print(f"  Claimed: 12% → 86.7% (+74.7%)")
            print(f"  Actual:  {baseline:.1f}% → {lora:.1f}% ({improvement:+.1f}%)")
    
    if 'student_prompt_only' in results:
        prompt_only = results['student_prompt_only']['accuracy'] * 100
        print(f"\nStage 4 - Prompt Only:        {prompt_only:.1f}%")
    
    print()
    print("="*80)
    print(f"✅ Results saved to: {CONFIG['results_dir']}/")
    print(f"📝 Verbose log: {CONFIG['results_dir']}/verbose_log.txt")
    print()


if __name__ == "__main__":
    main()