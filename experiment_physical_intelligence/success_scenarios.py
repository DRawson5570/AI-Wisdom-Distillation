#!/usr/bin/env python3
"""
Generate success scenarios where robot manipulations work correctly.
These are balanced against failure scenarios for proper evaluation.
"""

import json
import random
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class SuccessScenario:
    """A scenario where the manipulation succeeds"""
    scenario_id: int
    object_name: str
    attempted_action: str
    initial_strategy: Dict
    success_description: str
    environmental_factors: List[str]
    physical_principles_followed: str

def generate_success_scenarios(num_scenarios: int = 15, seed: int = 42) -> List[SuccessScenario]:
    """Generate diverse success scenarios"""
    random.seed(seed)
    scenarios = []
    scenario_id = 1000  # Start at 1000 to avoid confusion with failure IDs
    
    # Success templates
    templates = [
        # Proper grip and movement
        {
            "object": "soda_can",
            "action": "Grasp soda_can and lift vertically",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 8.5,
                "movement_speed": "slow",
                "reasoning": "Appropriate force for container weight, slow to minimize sloshing"
            },
            "description": "Object lifted smoothly without slipping. Slow speed prevented liquid sloshing.",
            "env": ["Container 40% full", "Dry surface"],
            "principle": "Used appropriate grip force (8.5N > minimum threshold) and slow movement to maintain stability."
        },
        {
            "object": "eggs",
            "action": "Pick up eggs carefully",
            "strategy": {
                "grip_type": "precision_grip",
                "grip_force_newtons": 2.0,
                "movement_speed": "slow",
                "reasoning": "Gentle grip for fragile object"
            },
            "description": "Eggs picked up successfully without damage. Gentle grip prevented crushing.",
            "env": ["Room temperature", "Stable surface"],
            "principle": "Applied low force (2N) appropriate for fragile objects, avoiding structural damage."
        },
        {
            "object": "milk_jug",
            "action": "Transfer milk_jug to shelf",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 12.0,
                "movement_speed": "moderate",
                "reasoning": "Strong grip with moderate speed for heavy liquid container"
            },
            "description": "Milk jug transferred successfully. Strong grip maintained stability throughout movement.",
            "env": ["Container 50% full", "Clear path"],
            "principle": "High grip force (12N) for heavy object, moderate speed to balance efficiency and stability."
        },
        {
            "object": "rice_bag",
            "action": "Lift rice_bag and place on counter",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 15.0,
                "movement_speed": "moderate",
                "reasoning": "Strong grip for heavy granular contents"
            },
            "description": "Rice bag lifted and placed successfully. High grip force prevented slipping.",
            "env": ["Dry packaging", "Stable grip point"],
            "principle": "Used sufficient force (15N) for heavy load, preventing grip failure."
        },
        {
            "object": "flour_bag",
            "action": "Move flour_bag to storage",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 10.0,
                "movement_speed": "slow",
                "reasoning": "Balanced force for moderately heavy bag"
            },
            "description": "Flour bag moved without incident. Controlled movement prevented shifting contents.",
            "env": ["Packaging intact", "Clear workspace"],
            "principle": "Applied adequate force with slow movement to maintain load stability."
        },
        {
            "object": "glass_jar",
            "action": "Pick up glass_jar carefully",
            "strategy": {
                "grip_type": "precision_grip",
                "grip_force_newtons": 6.0,
                "movement_speed": "slow",
                "reasoning": "Moderate force for fragile glass container"
            },
            "description": "Glass jar picked up successfully. Balanced force avoided both slipping and crushing.",
            "env": ["Clean surface", "Good visibility"],
            "principle": "Used moderate force (6N) sufficient for grip without exceeding material strength."
        },
        {
            "object": "soda_can",
            "action": "Place soda_can on level surface",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 7.0,
                "movement_speed": "slow",
                "reasoning": "Gentle placement to avoid tipping"
            },
            "description": "Can placed successfully on flat surface. Center of mass aligned vertically.",
            "env": ["Level surface", "Adequate clearance"],
            "principle": "Ensured geometric stability by aligning object's center of mass over base of support."
        },
        {
            "object": "juice_bottle",
            "action": "Transfer juice_bottle without spilling",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 9.0,
                "movement_speed": "slow",
                "reasoning": "Slow movement to prevent liquid sloshing"
            },
            "description": "Juice bottle transferred successfully. Slow acceleration minimized internal forces.",
            "env": ["Container 60% full", "Smooth motion path"],
            "principle": "Reduced acceleration to minimize inertial forces on liquid contents."
        },
        {
            "object": "strawberries",
            "action": "Pick up strawberries gently",
            "strategy": {
                "grip_type": "precision_grip",
                "grip_force_newtons": 1.5,
                "movement_speed": "slow",
                "reasoning": "Very gentle grip for soft fruit"
            },
            "description": "Strawberries picked up without bruising. Minimal force preserved fruit integrity.",
            "env": ["Fresh fruit", "Clean gripper"],
            "principle": "Applied minimal force (<2N) to avoid deforming soft biological material."
        },
        {
            "object": "can_soup",
            "action": "Grasp can_soup and lift",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 8.0,
                "movement_speed": "moderate",
                "reasoning": "Standard grip for cylindrical metal container"
            },
            "description": "Soup can lifted smoothly. Cylindrical shape provided stable grip surface.",
            "env": ["Dry can surface", "Good contact area"],
            "principle": "Exploited cylindrical geometry for optimal grip distribution."
        },
        {
            "object": "potato",
            "action": "Pick up potato from counter",
            "strategy": {
                "grip_type": "precision_grip",
                "grip_force_newtons": 4.0,
                "movement_speed": "moderate",
                "reasoning": "Moderate force for irregular shape"
            },
            "description": "Potato picked up successfully. Irregular shape accommodated by grip adjustment.",
            "env": ["Stable surface", "Good lighting"],
            "principle": "Adapted grip to irregular geometry while maintaining sufficient normal force."
        },
        {
            "object": "banana_bunch",
            "action": "Lift banana_bunch carefully",
            "strategy": {
                "grip_type": "precision_grip",
                "grip_force_newtons": 3.0,
                "movement_speed": "slow",
                "reasoning": "Gentle grip for soft fruit cluster"
            },
            "description": "Bananas lifted without damage. Low force prevented bruising soft peel.",
            "env": ["Room temperature", "Undamaged fruit"],
            "principle": "Applied gentle force appropriate for soft organic material susceptibility."
        },
        {
            "object": "milk_jug",
            "action": "Rotate milk_jug slowly",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 11.0,
                "movement_speed": "slow",
                "reasoning": "Slow rotation to prevent liquid sloshing"
            },
            "description": "Milk jug rotated successfully. Slow angular velocity prevented contents from shifting.",
            "env": ["Container 45% full", "Secure grip"],
            "principle": "Minimized angular acceleration to reduce centrifugal and Coriolis forces on liquid."
        },
        {
            "object": "rice_bag",
            "action": "Slide rice_bag along counter",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 13.0,
                "movement_speed": "slow",
                "reasoning": "Push with sufficient downward force"
            },
            "description": "Rice bag slid smoothly. Downward force overcame friction without lifting.",
            "env": ["Clean counter surface", "Adequate friction"],
            "principle": "Applied normal force exceeding friction threshold (F_grip > μ × m × g) for controlled sliding."
        },
        {
            "object": "soda_can",
            "action": "Navigate soda_can around obstacles",
            "strategy": {
                "grip_type": "power_grip",
                "grip_force_newtons": 7.5,
                "movement_speed": "slow",
                "reasoning": "Careful path planning with slow movement"
            },
            "description": "Can navigated successfully without collision. Slow speed allowed precise positioning.",
            "env": ["Cluttered workspace", "Adequate clearance maintained"],
            "principle": "Reduced velocity to increase reaction time and enable collision avoidance."
        }
    ]
    
    # Generate scenarios
    for template in templates[:num_scenarios]:
        scenario = SuccessScenario(
            scenario_id=scenario_id,
            object_name=template["object"],
            attempted_action=template["action"],
            initial_strategy=template["strategy"],
            success_description=template["description"],
            environmental_factors=template["env"],
            physical_principles_followed=template["principle"]
        )
        scenarios.append(scenario)
        scenario_id += 1
    
    return scenarios

def save_success_scenarios(scenarios: List[SuccessScenario], filename: str = "success_scenarios_dataset.json"):
    """Save scenarios to JSON file"""
    data = [asdict(scenario) for scenario in scenarios]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {len(scenarios)} success scenarios to {filename}")

if __name__ == "__main__":
    scenarios = generate_success_scenarios(num_scenarios=15)
    save_success_scenarios(scenarios)
    
    print(f"\n✅ Generated {len(scenarios)} success scenarios")
    print("\nSample scenario:")
    sample = scenarios[0]
    print(f"  Object: {sample.object_name}")
    print(f"  Action: {sample.attempted_action}")
    print(f"  Outcome: {sample.success_description}")
