import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Tuple, List, Optional

# --- Enums & Dataclasses ---
class Element(Enum):
    FIRE = auto()
    WATER = auto()
    WIND = auto()
    EARTH = auto()
    LIGHT = auto()
    DARK = auto()
    NEUTRAL = auto()

class HitType(Enum):
    MISS = auto()
    GLANCING = auto()
    NORMAL = auto()
    CRITICAL = auto()

@dataclass
class DamageBreakdownResult:
    is_hit: bool
    hit_type: HitType
    base_damage: float
    mitigation_model_used: str
    mitigated_damage: float
    elemental_multiplier: float
    total_damage: float
    can_proc_status: bool

# --- Elemental System ---
class ElementalSystem:
    # Definisi Matrix Kelemahan & Kekuatan Elemen
    _matrix: Dict[Element, Dict[Element, float]] = {
        Element.FIRE: {Element.WIND: 1.5, Element.WATER: 0.75},
        Element.WATER: {Element.FIRE: 1.5, Element.EARTH: 0.75},
        Element.WIND: {Element.EARTH: 1.5, Element.FIRE: 0.75},
        Element.EARTH: {Element.WATER: 1.5, Element.WIND: 0.75},
        Element.LIGHT: {Element.DARK: 1.5},
        Element.DARK: {Element.LIGHT: 1.5},
        Element.NEUTRAL: {}
    }

    @classmethod
    def get_multiplier(cls, attacker_element: Element, target_element: Element) -> float:
        if attacker_element == target_element:
            return 1.0
        
        target_dict = cls._matrix.get(attacker_element, {})
        if target_element in target_dict:
            return target_dict[target_element]
        
        # Cek jika target justru kuat terhadap attacker (misal API vs AIR -> AIR punya 1.5 thd API, 
        # maka API hanya berikan 0.75 dmg ke AIR)
        reverse_dict = cls._matrix.get(target_element, {})
        if attacker_element in reverse_dict and reverse_dict[attacker_element] == 1.5:
            return 0.75
            
        return 1.0

# --- Defense Mitigation Models ---
class MitigationModels:
    @staticmethod
    def linear_subtraction(atk: float, dfn: float, def_factor: float = 0.5, min_damage: float = 1.0) -> float:
        """Model A: Klasik RPG. Rentan stat inflation dan stalemate jika DEF > ATK."""
        return max(min_damage, atk - (dfn * def_factor))

    @staticmethod
    def hyperbolic(atk: float, dfn: float, defense_constant: float = 500.0) -> float:
        """Model B: JRPG/MOBA (League of Legends, dsb). Diminishing returns pada armor."""
        if dfn < 0: dfn = 0
        mitigation_ratio = defense_constant / (defense_constant + dfn)
        return atk * mitigation_ratio

    @staticmethod
    def percentage_threshold(atk: float, dfn: float, armor_efficiency: float = 0.002, max_mitigation: float = 0.80) -> float:
        """Model C: Berbasis % dengan hard cap tertentu untuk menjamin selalu ada damage yang masuk."""
        mitigation_percent = min(max_mitigation, dfn * armor_efficiency)
        return atk * (1.0 - mitigation_percent)

# --- Buff/Debuff Stacking ---
class StackingFormulas:
    @staticmethod
    def additive_stacking(base_value: float, buffs: List[float], debuffs: List[float] = None) -> float:
        if debuffs is None: debuffs = []
        total_modifier = sum(buffs) - sum(debuffs)
        return base_value * (1.0 + total_modifier)

    @staticmethod
    def multiplicative_stacking(base_value: float, buffs: List[float], debuffs: List[float] = None) -> float:
        if debuffs is None: debuffs = []
        result = base_value
        for b in buffs:
            result *= (1.0 + b)
        for d in debuffs:
            result *= (1.0 - d) 
        return result

# --- DoT Calculation ---
class DoTFormulas:
    @staticmethod
    def calculate_dot_damage(caster_atk: float, target_max_hp: float, atk_scaling: float = 0.5, hp_scaling: float = 0.05) -> float:
        """DoT biasanya kombinasi flat value dari %ATK caster + persentase dari Max HP korban."""
        return (caster_atk * atk_scaling) + (target_max_hp * hp_scaling)

# --- Core Combat Resolution Engine ---
def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(value, max_val))

def calculate_final_damage(
    attacker_atk: float,
    target_def: float,
    attacker_acc: float = 1.0,
    target_eva: float = 0.0,
    attacker_crit_rate: float = 0.1,
    attacker_crit_dmg: float = 1.5,
    target_crit_resist: float = 0.0,
    attacker_element: Element = Element.NEUTRAL,
    target_element: Element = Element.NEUTRAL,
    glancing_threshold: float = 0.15,
    min_hit_chance: float = 0.05,
    mitigation_model_func: Callable[[float, float], float] = lambda a, d: MitigationModels.hyperbolic(a, d, 500.0)
) -> DamageBreakdownResult:
    """Fungsi interface resolusi hit & kalkulasi total iterasi turn secara mendetail."""
    
    # 1. Hit Resolution (Akurasi vs Evasion)
    hit_chance = clamp(attacker_acc - target_eva, min_hit_chance, 1.0)
    roll = random.random()
    
    is_hit = False
    hit_type = HitType.MISS
    can_proc_status = False
    
    if roll > hit_chance:
        return DamageBreakdownResult(False, HitType.MISS, attacker_atk, mitigation_model_func.__name__, 0.0, 1.0, 0.0, False)
    
    is_hit = True
    
    # Check Glancing (Misal roll jatuh di ujung peluang miss)
    if roll > (hit_chance - glancing_threshold):
        hit_type = HitType.GLANCING
    else:
        # Check Crit
        crit_chance = clamp(attacker_crit_rate - target_crit_resist, 0.0, 1.0)
        crit_roll = random.random()
        if crit_roll <= crit_chance:
            hit_type = HitType.CRITICAL
            can_proc_status = True
        else:
            hit_type = HitType.NORMAL
            can_proc_status = True
            
    # 2. Base Mitigation
    mitigated_dmg = mitigation_model_func(attacker_atk, target_def)
    
    # 3. Elemental Multiplier
    elem_mult = ElementalSystem.get_multiplier(attacker_element, target_element)
    
    # 4. Total Multipliers Application
    final_dmg = mitigated_dmg * elem_mult
    
    if hit_type == HitType.CRITICAL:
        final_dmg *= attacker_crit_dmg
    elif hit_type == HitType.GLANCING:
        final_dmg *= 0.5  # Glancing hit = 50% damage penalty
        
    return DamageBreakdownResult(
        is_hit=True,
        hit_type=hit_type,
        base_damage=attacker_atk,
        mitigation_model_used=mitigation_model_func.__name__,
        mitigated_damage=mitigated_dmg,
        elemental_multiplier=elem_mult,
        total_damage=final_dmg,
        can_proc_status=can_proc_status
    )

# --- Sandbox Benchmarking ---
class FormulaSandbox:
    @staticmethod
    def compare_mitigation_curves(atk_value: float, def_range: List[float]):
        print(f"\n--- Mitigation Curves Comparison (Base ATK: {atk_value}) ---")
        print(f"{'DEF':<8} | {'Linear (f=0.5)':<15} | {'Hyperbolic (k=500)':<18} | {'Threshold (m=80%)':<18}")
        print("-" * 65)
        
        for dfn in def_range:
            lin = MitigationModels.linear_subtraction(atk_value, dfn, def_factor=0.5)
            hyp = MitigationModels.hyperbolic(atk_value, dfn, defense_constant=500.0)
            thr = MitigationModels.percentage_threshold(atk_value, dfn, armor_efficiency=0.002, max_mitigation=0.80)
            
            print(f"{dfn:<8} | {lin:<15.2f} | {hyp:<18.2f} | {thr:<18.2f}")
            
            # Anomaly Detection Output
            FormulaSandbox._detect_anomaly(atk_value, dfn, lin, hyp, thr)
            
    @staticmethod
    def _detect_anomaly(atk: float, dfn: float, lin: float, hyp: float, thr: float):
        # A simple anomaly check
        if lin <= 1.0 and atk > 10 and dfn > 0:
            print(f"  [!] Anomaly in Linear: Stalemated completely at DEF {dfn}. Attack is fully negated.")
        if thr == (atk * 0.2) and dfn > 0:
            print(f"  [*] Notice: Threshold hard-cap (80% mitigate) reached at DEF {dfn}")

if __name__ == "__main__":
    def wrap_linear(a, d): return MitigationModels.linear_subtraction(a, d)
    def wrap_hyperbolic(a, d): return MitigationModels.hyperbolic(a, d)
    def wrap_threshold(a, d): return MitigationModels.percentage_threshold(a, d)
    
    wrap_linear.__name__ = "Linear Subtraction"
    wrap_hyperbolic.__name__ = "Hyperbolic Curve"
    wrap_threshold.__name__ = "Percentage Tiered"

    print("=== RPG Damage Formula & Combat Logic Library ===\n")
    
    # 1. Compare Mitigation Curves (Benchmarking)
    def_test_range = [0, 50, 100, 250, 400, 500, 1000, 2500]
    FormulaSandbox.compare_mitigation_curves(atk_value=1000, def_range=def_test_range)
    
    # 2. Detailed Single Attack Resolution
    print("\n\n--- Detailed Attack Simulation (Fire vs Wind) ---")
    
    # Fix seed to ensure reproducible demo output
    random.seed(42)
    
    result = calculate_final_damage(
        attacker_atk=1500.0,
        target_def=600.0,
        attacker_acc=0.90,
        target_eva=0.10,
        attacker_crit_rate=0.60,
        attacker_crit_dmg=2.5,
        target_crit_resist=0.10,
        attacker_element=Element.FIRE,
        target_element=Element.WIND,
        mitigation_model_func=wrap_hyperbolic
    )
    
    print(f"Attacker Hit Check Result : {'Success' if result.is_hit else 'Failed'}")
    if result.is_hit:
        print(f"Hit Type                  : {result.hit_type.name}")
        print(f"Base ATK Damage           : {result.base_damage:.2f}")
        print(f"Mitigation Model          : {result.mitigation_model_used}")
        print(f"Damage After DEF          : {result.mitigated_damage:.2f} (-{(1 - (result.mitigated_damage/result.base_damage))*100:.1f}%)")
        print(f"Elemental Multiplier      : x{result.elemental_multiplier:.2f} (Advantage)")
        if result.hit_type == HitType.CRITICAL:
            print(f"Critical Multiplier       : x2.50")
        elif result.hit_type == HitType.GLANCING:
            print(f"Glancing Penalty          : x0.50")
        print(f"Status Effect Proc Status : {'Can Proc' if result.can_proc_status else 'Blocked'}")
        print(f"--> FINAL TOTAL DAMAGE    : {result.total_damage:.2f}")
