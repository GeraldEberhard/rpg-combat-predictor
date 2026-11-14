import random
import math
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any

class EffectType(Enum):
    BURN = auto()
    STUN = auto()
    ATK_BUFF = auto()
    DEF_DEBUFF = auto()

@dataclass
class Effect:
    effect_type: EffectType
    duration: int
    value: float
    proc_chance: float = 1.0

@dataclass
class Character:
    name: str
    max_hp: float
    current_hp: float
    atk: float
    defense: float
    speed: float
    accuracy: float
    crit_rate: float
    crit_dmg: float = 1.5
    active_effects: List[Effect] = field(default_factory=list)

    def apply_effect(self, effect: Effect):
        if random.random() <= effect.proc_chance:
            self.active_effects.append(copy.deepcopy(effect))

    def tick_effects(self) -> float:
        """Process start-of-turn effects like BURN and return direct damage taken, then decrement durations."""
        burn_damage = 0.0
        new_effects = []
        for effect in self.active_effects:
            if effect.effect_type == EffectType.BURN:
                burn_damage += effect.value
            
            effect.duration -= 1
            if effect.duration > 0:
                new_effects.append(effect)
        
        self.active_effects = new_effects
        self.current_hp = max(0.0, self.current_hp - burn_damage)
        return burn_damage

    def has_stun(self) -> bool:
        return any(e.effect_type == EffectType.STUN for e in self.active_effects)

    def get_effective_atk(self) -> float:
        atk_buff_sum = sum(e.value for e in self.active_effects if e.effect_type == EffectType.ATK_BUFF)
        return self.atk * (1.0 + atk_buff_sum)

    def get_effective_def(self) -> float:
        def_debuff_sum = sum(e.value for e in self.active_effects if e.effect_type == EffectType.DEF_DEBUFF)
        return self.defense * max(0.0, (1.0 - def_debuff_sum))

class DuelResult(Enum):
    WIN_A = auto()
    WIN_B = auto()
    DRAW = auto()

@dataclass
class DuelStats:
    winner: DuelResult
    rounds: int
    damage_dealt_a: float
    damage_dealt_b: float

def execute_attack(attacker: Character, defender: Character) -> float:
    """Executes a standard attack and returns the mitigated damage dealt."""
    # Hit Check
    if random.random() > attacker.accuracy:
        return 0.0  # Miss

    # Crit Check
    is_crit = random.random() <= attacker.crit_rate
    crit_mult = attacker.crit_dmg if is_crit else 1.0

    eff_atk = attacker.get_effective_atk()
    eff_def = defender.get_effective_def()

    # Damage Formula
    mitigated_damage = max(1.0, (eff_atk * crit_mult) - (eff_def * 0.5))
    
    defender.current_hp = max(0.0, defender.current_hp - mitigated_damage)
    return mitigated_damage

def simulate_duel(char_a: Character, char_b: Character, max_turns: int = 50) -> DuelStats:
    # Deepcopy to ensure simulation does not leak state
    a = copy.deepcopy(char_a)
    b = copy.deepcopy(char_b)
    
    a.current_hp = a.max_hp
    b.current_hp = b.max_hp
    a.active_effects = []
    b.active_effects = []

    # Determine Turn Order based on speed
    if a.speed > b.speed:
        first, second = a, b
    elif b.speed > a.speed:
        first, second = b, a
    else:
        # If speed is tied, coin toss
        first, second = (a, b) if random.random() < 0.5 else (b, a)

    total_dmg_a = 0.0
    total_dmg_b = 0.0
    current_round = 0

    while current_round < max_turns and a.current_hp > 0 and b.current_hp > 0:
        current_round += 1
        
        # --- First Character's Turn ---
        burn_dmg_first = first.tick_effects()
        if first == a:
            total_dmg_b += burn_dmg_first
        else:
            total_dmg_a += burn_dmg_first

        if first.current_hp <= 0:
            break

        if not first.has_stun():
            dmg = execute_attack(first, second)
            if first == a:
                total_dmg_a += dmg
            else:
                total_dmg_b += dmg
        
        if second.current_hp <= 0:
            break

        # --- Second Character's Turn ---
        burn_dmg_second = second.tick_effects()
        if second == a:
            total_dmg_b += burn_dmg_second
        else:
            total_dmg_a += burn_dmg_second

        if second.current_hp <= 0:
            break

        if not second.has_stun():
            dmg = execute_attack(second, first)
            if second == a:
                total_dmg_a += dmg
            else:
                total_dmg_b += dmg
            
    # Determine winner
    if a.current_hp > 0 and b.current_hp <= 0:
        winner = DuelResult.WIN_A
    elif b.current_hp > 0 and a.current_hp <= 0:
        winner = DuelResult.WIN_B
    else:
        winner = DuelResult.DRAW

    return DuelStats(winner, current_round, total_dmg_a, total_dmg_b)

def run_monte_carlo_duel(char_a: Character, char_b: Character, n_simulations: int = 1000) -> Dict[str, Any]:
    wins_a = 0
    wins_b = 0
    draws = 0
    total_rounds = 0
    
    dmg_a_history = []
    dmg_b_history = []
    
    for _ in range(n_simulations):
        stats = simulate_duel(char_a, char_b)
        
        if stats.winner == DuelResult.WIN_A:
            wins_a += 1
        elif stats.winner == DuelResult.WIN_B:
            wins_b += 1
        else:
            draws += 1
            
        total_rounds += stats.rounds
        dmg_a_history.append(stats.damage_dealt_a)
        dmg_b_history.append(stats.damage_dealt_b)

    def calc_stats(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "std_dev": 0.0}
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std_dev = math.sqrt(variance)
        return {
            "min": min(data),
            "max": max(data),
            "avg": mean,
            "std_dev": std_dev
        }

    return {
        "n_simulations": n_simulations,
        "win_rate_a": (wins_a / n_simulations) * 100.0,
        "win_rate_b": (wins_b / n_simulations) * 100.0,
        "draw_rate": (draws / n_simulations) * 100.0,
        "average_rounds": total_rounds / n_simulations,
        "average_total_damage_a": sum(dmg_a_history) / n_simulations,
        "average_total_damage_b": sum(dmg_b_history) / n_simulations,
        "damage_distribution_a": calc_stats(dmg_a_history),
        "damage_distribution_b": calc_stats(dmg_b_history)
    }

if __name__ == "__main__":
    # Test characters
    warrior = Character(
        name="Warrior",
        max_hp=1200,
        current_hp=1200,
        atk=180,
        defense=120,
        speed=10,
        accuracy=0.95,
        crit_rate=0.15,
        crit_dmg=1.5
    )
    
    assassin = Character(
        name="Assassin",
        max_hp=800,
        current_hp=800,
        atk=220,
        defense=70,
        speed=25,
        accuracy=0.90,
        crit_rate=0.40,
        crit_dmg=2.0
    )

    print(f"Starting Monte Carlo Simulation ({warrior.name} vs {assassin.name})...")
    
    simulations = 10000
    results = run_monte_carlo_duel(warrior, assassin, n_simulations=simulations)
    
    print(f"\n--- Monte Carlo Results ({results['n_simulations']} simulations) ---")
    print(f"Win Rate: {warrior.name} = {results['win_rate_a']:.2f}% | {assassin.name} = {results['win_rate_b']:.2f}% (Draws: {results['draw_rate']:.2f}%)")
    print(f"Average Rounds (Turn-to-Kill): {results['average_rounds']:.2f}")
    
    dmg_a = results['damage_distribution_a']
    dmg_b = results['damage_distribution_b']
    
    print(f"\nDamage Dealt by {warrior.name}:")
    print(f"  Avg: {dmg_a['avg']:.2f}, Min: {dmg_a['min']:.2f}, Max: {dmg_a['max']:.2f}, StdDev: {dmg_a['std_dev']:.2f}")
    
    print(f"Damage Dealt by {assassin.name}:")
    print(f"  Avg: {dmg_b['avg']:.2f}, Min: {dmg_b['min']:.2f}, Max: {dmg_b['max']:.2f}, StdDev: {dmg_b['std_dev']:.2f}")

    # Adding a simple effect test
    print("\n--- Testing With Status Effects ---")
    warrior_with_effect = copy.deepcopy(warrior)
    # Give warrior an ATK buff of 50% for 3 turns
    warrior_with_effect.apply_effect(Effect(EffectType.ATK_BUFF, duration=3, value=0.5, proc_chance=1.0))
    
    assassin_with_effect = copy.deepcopy(assassin)
    # Assassin starts with a stun status for 1 turn (Stun skips the first turn)
    assassin_with_effect.apply_effect(Effect(EffectType.STUN, duration=1, value=0, proc_chance=1.0))
    
    results_effects = run_monte_carlo_duel(warrior_with_effect, assassin_with_effect, n_simulations=5000)
    print(f"Win Rate (Warrior Buffed vs Assassin Stunned): {warrior.name} = {results_effects['win_rate_a']:.2f}% | {assassin.name} = {results_effects['win_rate_b']:.2f}%")
