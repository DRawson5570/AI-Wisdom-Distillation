#!/usr/bin/env python3
"""
Claude Self-Improvement Experiment
===================================

Pure LRL experiment showing Claude learning through reflection.

Stages:
1. Claude Baseline (no learning)
2. Claude LRL Training (learns from mistakes)
3. Claude Post-Learning Test (applies learned strategy)

No student model, no LoRA - just watching Claude get smarter!
"""

import os
import sys

# Add parent directory to path to import framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from framework import (
    TeacherModel,
    run_knowledge_transfer
)
from meeting_scheduling_lrl import MeetingSchedulingDomain

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

CONFIG = {
    # Model
    'model': 'claude-3-5-haiku-20241022',
    
    # Dataset sizes
    'num_train': 250,  # Training problems for LRL
    'num_test': 150,   # Test set size
    
    # Output
    'results_dir': './results_claude_self_improvement',
    'verbose': True,
    
    # Stages to run
    'run_baseline': True,
    'run_learning': True,
    'run_post_test': True,
}


def main():
    """Run Claude self-improvement experiment"""
    
    print("="*80)
    print("CLAUDE SELF-IMPROVEMENT EXPERIMENT")
    print("="*80)
    print()
    print("Watching Claude learn through pure LRL:")
    print("  Stage 1: Baseline performance (no learning)")
    print("  Stage 2: LRL Training (learns from mistakes)")
    print("  Stage 3: Post-learning test (applies strategy)")
    print()
    print("Configuration:")
    print(f"  Model: {CONFIG['model']}")
    print(f"  Training: {CONFIG['num_train']} problems")
    print(f"  Testing: {CONFIG['num_test']} problems")
    print()
    print("="*80)
    print()
    
    # Create domain
    domain = MeetingSchedulingDomain()
    
    # Get API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠️  Error: ANTHROPIC_API_KEY not set!")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
        return
    
    # Generate problems
    print("📋 Generating problems...")
    train_problems = domain.generate_problems(CONFIG['num_train'], seed=42)
    test_problems = domain.generate_problems(CONFIG['num_test'], seed=123)
    print(f"   Training: {len(train_problems)} problems")
    print(f"   Testing: {len(test_problems)} problems")
    print()
    
    # Import what we need for evaluation
    from framework.knowledge_transfer import evaluate_model
    from pathlib import Path
    import json
    
    # Setup output directory
    output_dir = Path(CONFIG['results_dir'])
    
    # Archive previous results if they exist
    if output_dir.exists():
        from datetime import datetime
        import shutil
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = output_dir.parent / 'archive'
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"{output_dir.name}_{timestamp}"
        shutil.move(str(output_dir), str(archive_path))
        print(f"📦 Previous results archived to: {archive_path}/")
        print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "verbose_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Claude Self-Improvement Experiment\n")
        f.write(f"Model: {CONFIG['model']}\n")
        f.write("="*80 + "\n\n")
    
    results = {}
    
    # ========================================================================
    # STAGE 1: Baseline
    # ========================================================================
    if CONFIG['run_baseline']:
        print("\n" + "="*80)
        print("🧪 STAGE 1: Claude Baseline (No Learning)")
        print("="*80)
        print()
        
        claude_baseline = TeacherModel(
            CONFIG['model'], 
            api_key, 
            enable_learning=False,
            use_ollama=False
        )
        
        results['baseline'] = evaluate_model(
            claude_baseline, 
            test_problems, 
            domain, 
            "Stage 1: Claude Baseline",
            verbose_file=CONFIG['verbose'],
            verbose_terminal=False,
            log_file=log_file
        )
        
        print(f"\n✅ Baseline: {results['baseline']['accuracy']:.1%}")
    
    # ========================================================================
    # STAGE 2: LRL Training
    # ========================================================================
    if CONFIG['run_learning']:
        print("\n" + "="*80)
        print("🧪 STAGE 2: Claude LRL Training")
        print("="*80)
        print()
        
        claude_learning = TeacherModel(
            CONFIG['model'], 
            api_key, 
            enable_learning=True,
            use_ollama=False
        )
        
        results['learning'] = evaluate_model(
            claude_learning, 
            train_problems, 
            domain, 
            "Stage 2: Claude Learning",
            verbose_file=CONFIG['verbose'],
            verbose_terminal=False,
            log_file=log_file
        )
        
        # Save strategy and journal
        learned_strategy = claude_learning.get_strategy()
        (output_dir / "learned_strategy.txt").write_text(learned_strategy)
        (output_dir / "learning_journal.txt").write_text(claude_learning.journal)
        
        print(f"\n✅ Training complete: {results['learning']['accuracy']:.1%}")
        print(f"💾 Strategy saved to: {output_dir}/learned_strategy.txt")
        print(f"📝 Journal saved to: {output_dir}/learning_journal.txt")
    
    # ========================================================================
    # STAGE 3: Post-Learning Test
    # ========================================================================
    if CONFIG['run_post_test']:
        print("\n" + "="*80)
        print("🧪 STAGE 3: Claude Post-Learning Test")
        print("="*80)
        print()
        
        # Load the learned strategy
        strategy_file = output_dir / "learned_strategy.txt"
        if not strategy_file.exists():
            print("⚠️  No strategy found. Run learning stage first!")
            return
        
        learned_strategy = strategy_file.read_text()
        
        # Create new Claude instance with learning disabled but strategy loaded
        claude_post = TeacherModel(
            CONFIG['model'], 
            api_key, 
            enable_learning=False,
            use_ollama=False
        )
        # Inject the learned strategy
        claude_post.strategy = learned_strategy
        
        results['post_learning'] = evaluate_model(
            claude_post, 
            test_problems, 
            domain, 
            "Stage 3: Claude Post-Learning",
            strategy=learned_strategy,  # Pass the learned strategy!
            verbose_file=CONFIG['verbose'],
            verbose_terminal=False,
            log_file=log_file
        )
        
        print(f"\n✅ Post-learning: {results['post_learning']['accuracy']:.1%}")
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("📊 CLAUDE SELF-IMPROVEMENT RESULTS")
    print("="*80)
    print()
    
    if 'baseline' in results:
        baseline = results['baseline']['accuracy'] * 100
        print(f"Stage 1 - Baseline (no learning):  {baseline:.1f}%")
    
    if 'learning' in results:
        learning = results['learning']['accuracy'] * 100
        print(f"Stage 2 - During LRL training:    {learning:.1f}%")
    
    if 'post_learning' in results:
        post = results['post_learning']['accuracy'] * 100
        print(f"Stage 3 - After learning:          {post:.1f}%")
        
        if 'baseline' in results:
            improvement = post - baseline
            print()
            print(f"📈 Improvement: {baseline:.1f}% → {post:.1f}% ({improvement:+.1f}%)")
            
            if improvement > 0:
                print(f"   ✅ Claude improved by {improvement:.1f} percentage points!")
            elif improvement < 0:
                print(f"   ⚠️  Claude got worse by {abs(improvement):.1f} percentage points")
            else:
                print(f"   ➡️  No change in performance")
    
    print()
    print("="*80)
    
    # Save results
    results_file = output_dir / "results.json"
    results_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"💾 Results saved to: {results_file}")
    print(f"📝 Full log: {log_file}")
    print()


if __name__ == "__main__":
    main()
