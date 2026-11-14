import os
import random
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, Any

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, roc_auc_score
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier

# Impor engine simulator yang telah dibuat sebelumnya
from combat_engine import Character, Effect, EffectType, run_monte_carlo_duel

warnings.filterwarnings("ignore")

def generate_random_character(name: str) -> Character:
    """Membuat karakter dengan status yang digenerate secara acak (random sampling)."""
    char = Character(
        name=name,
        max_hp=random.uniform(500, 3000),
        current_hp=0, # Akan diset ke max_hp setelahnya
        atk=random.uniform(50, 500),
        defense=random.uniform(20, 250),
        speed=random.uniform(50, 200),
        accuracy=random.uniform(0.60, 1.00),
        crit_rate=random.uniform(0.05, 0.75),
        crit_dmg=random.uniform(1.2, 2.5)
    )
    char.current_hp = char.max_hp
    
    # Assign efek status acak (0 hingga 2 efek) pada start-up
    num_effects = random.randint(0, 2)
    effect_types = list(EffectType)
    
    for _ in range(num_effects):
        etype = random.choice(effect_types)
        duration = random.randint(1, 3)
        if etype == EffectType.STUN:
            value = 0.0
        elif etype == EffectType.BURN:
            value = random.uniform(50, 200)
        else:
            value = random.uniform(0.1, 0.5)
        
        proc_chance = random.uniform(0.5, 1.0)
        char.apply_effect(Effect(etype, duration, value, proc_chance))
        
    return char

def compute_features(char_a: Character, char_b: Character) -> Dict[str, float]:
    """Feature Engineering dari interaksi dua karakter."""
    hp_ratio = char_a.max_hp / char_b.max_hp if char_b.max_hp > 0 else 1.0
    atk_to_def_ratio = char_a.atk / char_b.defense if char_b.defense > 0 else char_a.atk
    speed_advantage = char_a.speed - char_b.speed
    
    # Estimasi DPS berbasis peluang kritikal dan akurasi
    eff_dps_a = char_a.atk * char_a.accuracy * (1 + char_a.crit_rate * (char_a.crit_dmg - 1))
    eff_dps_b = char_b.atk * char_b.accuracy * (1 + char_b.crit_rate * (char_b.crit_dmg - 1))
    
    # Hitung keunggulan Stun (kategorikal/numerik)
    has_stun_a = 1 if any(e.effect_type == EffectType.STUN for e in char_a.active_effects) else 0
    has_stun_b = 1 if any(e.effect_type == EffectType.STUN for e in char_b.active_effects) else 0
    has_stun_advantage = has_stun_a - has_stun_b
    
    # Kalkulasi potensi akumulasi DoT (Burn)
    burn_a = sum(e.value * e.proc_chance * e.duration for e in char_a.active_effects if e.effect_type == EffectType.BURN)
    burn_b = sum(e.value * e.proc_chance * e.duration for e in char_b.active_effects if e.effect_type == EffectType.BURN)
    dot_burn_potential = burn_a - burn_b

    return {
        # Raw features
        "a_hp": char_a.max_hp, "a_atk": char_a.atk, "a_def": char_a.defense, "a_spd": char_a.speed,
        "a_acc": char_a.accuracy, "a_crit": char_a.crit_rate, "a_cdmg": char_a.crit_dmg,
        "b_hp": char_b.max_hp, "b_atk": char_b.atk, "b_def": char_b.defense, "b_spd": char_b.speed,
        "b_acc": char_b.accuracy, "b_crit": char_b.crit_rate, "b_cdmg": char_b.crit_dmg,
        # Engineered relational features
        "hp_ratio": hp_ratio,
        "atk_to_def_ratio": atk_to_def_ratio,
        "speed_advantage": speed_advantage,
        "effective_dps_a": eff_dps_a,
        "effective_dps_b": eff_dps_b,
        "has_stun_advantage": has_stun_advantage,
        "dot_burn_potential": dot_burn_potential
    }

def generate_combat_dataset(n_samples: int = 2000, sims_per_matchup: int = 50) -> pd.DataFrame:
    """Generate kombinasi pertarungan secara random lalu jalankan simulasi monte carlo."""
    print(f"Generating synthetic dataset: {n_samples} samples with {sims_per_matchup} simulations per matchup...")
    data = []
    
    for i in range(n_samples):
        char_a = generate_random_character(f"CharA_{i}")
        char_b = generate_random_character(f"CharB_{i}")
        
        # Monte Carlo untuk memperoleh Ground Truth metrics
        sim_results = run_monte_carlo_duel(char_a, char_b, n_simulations=sims_per_matchup)
        
        # Ekstrak metrik
        win_prob_a = sim_results['win_rate_a'] / 100.0
        avg_dmg_a = sim_results['average_total_damage_a']
        
        features = compute_features(char_a, char_b)
        row = {**features, "target_damage_a": avg_dmg_a, "target_win_prob_a": win_prob_a}
        data.append(row)
        
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1} / {n_samples} samples.")
            
    df = pd.DataFrame(data)
    print("Dataset generation complete!")
    return df

def train_models(df: pd.DataFrame, model_dir: str = "./models"):
    """Latih model Machine Learning menggunakan dataset sintetis yang dihasilkan."""
    os.makedirs(model_dir, exist_ok=True)
    
    X = df.drop(columns=["target_damage_a", "target_win_prob_a"])
    y_reg = df["target_damage_a"]
    y_clf_prob = df["target_win_prob_a"]
    
    # Karena XGBClassifier butuh integer classes, kita konversi probabilitas menjadi biner untuk klasifikasi,
    # namun performanya tetap kita ukur dengan probabilitas menggunakan predict_proba
    y_class = (y_clf_prob > 0.5).astype(int)
    
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )

    print("\n--- Training Target 1 (Regression): LightGBM for Average Damage A ---")
    reg_param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9]
    }
    
    lgbm_base = LGBMRegressor(random_state=42, verbose=-1)
    reg_search = RandomizedSearchCV(lgbm_base, reg_param_dist, n_iter=10, cv=5, scoring='neg_root_mean_squared_error', random_state=42)
    reg_search.fit(X_train, y_reg_train)
    
    best_reg = reg_search.best_estimator_
    y_reg_pred = best_reg.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    
    print(f"Best Params: {reg_search.best_params_}")
    print(f"Evaluation RMSE: {rmse:.4f}")
    print(f"Evaluation MAE: {mae:.4f}")
    
    print("\n--- Training Target 2 (Classification): XGBoost for Win Probability A ---")
    clf_param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
    clf_search = RandomizedSearchCV(xgb_base, clf_param_dist, n_iter=10, cv=5, scoring='neg_log_loss', random_state=42)
    clf_search.fit(X_train, y_class_train)
    
    best_clf = clf_search.best_estimator_
    y_clf_pred_proba = best_clf.predict_proba(X_test)[:, 1]
    
    ll = log_loss(y_class_test, y_clf_pred_proba)
    roc = roc_auc_score(y_class_test, y_clf_pred_proba)
    
    print(f"Best Params: {clf_search.best_params_}")
    print(f"Evaluation Log Loss: {ll:.4f}")
    print(f"Evaluation ROC-AUC: {roc:.4f}")
    
    # Ekspor Model ke Disk
    reg_model_path = os.path.join(model_dir, "damage_regressor.pkl")
    clf_model_path = os.path.join(model_dir, "winrate_classifier.pkl")
    joblib.dump(best_reg, reg_model_path)
    joblib.dump(best_clf, clf_model_path)
    print(f"\nModels successfully saved to {model_dir}")
    
    return best_reg, best_clf

def predict_combat_outcome(char_a_stats: dict, char_b_stats: dict, model_dir: str = "./models") -> dict:
    """Fungsi inferensi instan untuk memprediksi damage dan win rate tanpa simulasi berat."""
    reg_model = joblib.load(os.path.join(model_dir, "damage_regressor.pkl"))
    clf_model = joblib.load(os.path.join(model_dir, "winrate_classifier.pkl"))
    
    def build_character(d: dict, name: str) -> Character:
        char = Character(
            name=name, max_hp=d['hp'], current_hp=d['hp'],
            atk=d['atk'], defense=d['def'], speed=d['spd'],
            accuracy=d['acc'], crit_rate=d['crit'], crit_dmg=d.get('cdmg', 1.5)
        )
        if 'effects' in d:
            for eff in d['effects']:
                char.apply_effect(Effect(eff['type'], eff['duration'], eff['value'], eff.get('chance', 1.0)))
        return char

    char_a = build_character(char_a_stats, "Char_A")
    char_b = build_character(char_b_stats, "Char_B")
    
    features = compute_features(char_a, char_b)
    df_features = pd.DataFrame([features])
    
    predicted_damage = reg_model.predict(df_features)[0]
    predicted_win_prob = clf_model.predict_proba(df_features)[0][1]
    
    return {
        "predicted_avg_damage_a": predicted_damage,
        "predicted_win_probability_a": predicted_win_prob
    }

if __name__ == "__main__":
    print("=== Combat ML Pipeline Start ===")
    # 1. Dataset Generation
    # Menggunakan 1000 sampel untuk demo, simulasi Monte Carlo 20 kali per sampel agar lebih cepat.
    df = generate_combat_dataset(n_samples=1000, sims_per_matchup=20)
    
    # 2. Model Training & Evaluation
    train_models(df)
    
    # 3. Model Inference Test
    print("\n=== Real-time Inference Simulation ===")
    test_char_a = {
        'hp': 1200, 'atk': 180, 'def': 120, 'spd': 100, 
        'acc': 0.95, 'crit': 0.15, 'cdmg': 1.5,
        'effects': [{'type': EffectType.ATK_BUFF, 'duration': 3, 'value': 0.5, 'chance': 1.0}]
    }
    
    test_char_b = {
        'hp': 800, 'atk': 220, 'def': 70, 'spd': 120, 
        'acc': 0.90, 'crit': 0.40, 'cdmg': 2.0,
        'effects': [{'type': EffectType.STUN, 'duration': 1, 'value': 0.0, 'chance': 1.0}]
    }
    
    outcome = predict_combat_outcome(test_char_a, test_char_b)
    print("Matchup: Warrior (Buffed) vs Assassin (Stunned)")
    print(f"Predicted Avg Damage by Warrior: {outcome['predicted_avg_damage_a']:.2f}")
    print(f"Predicted Win Probability for Warrior: {outcome['predicted_win_probability_a'] * 100:.2f}%")
    print("=== Pipeline Complete ===")
