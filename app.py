import os
import time
import copy
import joblib
import pandas as pd
import streamlit as st

# Import komponen dari combat_engine dan train_combat_ml
from combat_engine import Character, Effect, EffectType, run_monte_carlo_duel
from train_combat_ml import compute_features

st.set_page_config(
    page_title="RPG Combat ML Predictor",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2d313e;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .card-hero {
        border-left: 4px solid #3b82f6;
    }
    .card-enemy {
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reg_path = os.path.join(base_dir, "models", "damage_regressor.pkl")
    clf_path = os.path.join(base_dir, "models", "winrate_classifier.pkl")
    
    if not os.path.exists(reg_path) or not os.path.exists(clf_path):
        # Fallback jika folder models ada di current working directory
        reg_path = os.path.join("models", "damage_regressor.pkl")
        clf_path = os.path.join("models", "winrate_classifier.pkl")
        
    reg_model = joblib.load(reg_path)
    clf_model = joblib.load(clf_path)
    return reg_model, clf_model

try:
    reg_model, clf_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Gagal memuat model: {e}")

# --- Header ---
st.markdown('<div class="main-header">⚔️ RPG Combat Outcome Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dashboard Interaktif Inferensi Cepat Machine Learning (LightGBM & XGBoost) untuk Simulasi Duel 1v1</div>', unsafe_allow_html=True)

# --- Sidebar Presets ---
st.sidebar.title("🎮 Template Karakter")
preset_choice = st.sidebar.selectbox(
    "Pilih Preset Matchup:",
    [
        "Custom Matchup",
        "Warrior (Tanky) vs Assassin (Speed/Crit)",
        "Glass Cannon Mage vs Paladin (Heavy Armor)",
        "Boss Raid Titan vs Agi Rogue"
    ]
)

# Preset configs
presets = {
    "Warrior (Tanky) vs Assassin (Speed/Crit)": {
        "a": {"name": "Warrior", "hp": 1500, "atk": 220, "def": 150, "spd": 85, "acc": 0.95, "crit": 0.15, "cdmg": 1.5, "atk_buff": 0.20, "def_debuff": 0.0, "stun": False, "burn": 0},
        "b": {"name": "Assassin", "hp": 850, "atk": 310, "def": 65, "spd": 160, "acc": 0.90, "crit": 0.50, "cdmg": 2.2, "atk_buff": 0.0, "def_debuff": 0.0, "stun": False, "burn": 0}
    },
    "Glass Cannon Mage vs Paladin (Heavy Armor)": {
        "a": {"name": "Mage", "hp": 750, "atk": 420, "def": 40, "spd": 110, "acc": 0.95, "crit": 0.30, "cdmg": 1.8, "atk_buff": 0.30, "def_debuff": 0.0, "stun": False, "burn": 120},
        "b": {"name": "Paladin", "hp": 2200, "atk": 160, "def": 210, "spd": 70, "acc": 0.88, "crit": 0.10, "cdmg": 1.4, "atk_buff": 0.0, "def_debuff": 0.0, "stun": False, "burn": 0}
    },
    "Boss Raid Titan vs Agi Rogue": {
        "a": {"name": "Agi Rogue", "hp": 900, "atk": 250, "def": 80, "spd": 180, "acc": 0.98, "crit": 0.60, "cdmg": 2.5, "atk_buff": 0.0, "def_debuff": 0.25, "stun": True, "burn": 80},
        "b": {"name": "Boss Titan", "hp": 3000, "atk": 380, "def": 230, "spd": 55, "acc": 0.75, "crit": 0.15, "cdmg": 1.6, "atk_buff": 0.0, "def_debuff": 0.0, "stun": False, "burn": 0}
    }
}

active_preset = presets.get(preset_choice, None)

# --- Input Kolom Karakter ---
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🛡️ Karakter A (Attacker / Hero)")
    name_a = st.text_input("Nama Karakter A", value=active_preset["a"]["name"] if active_preset else "Hero Alpha")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        hp_a = st.slider("Max HP (A)", 500, 3500, int(active_preset["a"]["hp"]) if active_preset else 1200, step=50)
        atk_a = st.slider("ATK (A)", 50, 600, int(active_preset["a"]["atk"]) if active_preset else 200, step=10)
        def_a = st.slider("Defense (A)", 20, 300, int(active_preset["a"]["def"]) if active_preset else 100, step=5)
        spd_a = st.slider("Speed (A)", 50, 220, int(active_preset["a"]["spd"]) if active_preset else 100, step=5)
    with col_a2:
        acc_a = st.slider("Accuracy (A)", 0.60, 1.00, float(active_preset["a"]["acc"]) if active_preset else 0.95, step=0.01)
        crit_a = st.slider("Crit Rate (A)", 0.05, 0.90, float(active_preset["a"]["crit"]) if active_preset else 0.20, step=0.01)
        cdmg_a = st.slider("Crit DMG (A)", 1.2, 3.0, float(active_preset["a"]["cdmg"]) if active_preset else 1.5, step=0.1)

    with st.expander("✨ Status Effects & Buffs (Karakter A)"):
        buff_atk_a = st.slider("ATK Buff (%)", 0, 100, int(active_preset["a"]["atk_buff"] * 100) if active_preset else 0, step=5) / 100.0
        debuff_def_a = st.slider("DEF Debuff (%)", 0, 100, int(active_preset["a"]["def_debuff"] * 100) if active_preset else 0, step=5) / 100.0
        stun_a = st.checkbox("Mulai dalam keadaan STUN?", value=active_preset["a"]["stun"] if active_preset else False)
        burn_a = st.number_input("DoT Burn Damage/turn (A)", min_value=0, max_value=500, value=int(active_preset["a"]["burn"]) if active_preset else 0, step=20)

with col_b:
    st.markdown("### ⚔️ Karakter B (Defender / Opponent)")
    name_b = st.text_input("Nama Karakter B", value=active_preset["b"]["name"] if active_preset else "Enemy Beta")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        hp_b = st.slider("Max HP (B)", 500, 3500, int(active_preset["b"]["hp"]) if active_preset else 1000, step=50)
        atk_b = st.slider("ATK (B)", 50, 600, int(active_preset["b"]["atk"]) if active_preset else 210, step=10)
        def_b = st.slider("Defense (B)", 20, 300, int(active_preset["b"]["def"]) if active_preset else 90, step=5)
        spd_b = st.slider("Speed (B)", 50, 220, int(active_preset["b"]["spd"]) if active_preset else 110, step=5)
    with col_b2:
        acc_b = st.slider("Accuracy (B)", 0.60, 1.00, float(active_preset["b"]["acc"]) if active_preset else 0.90, step=0.01)
        crit_b = st.slider("Crit Rate (B)", 0.05, 0.90, float(active_preset["b"]["crit"]) if active_preset else 0.25, step=0.01)
        cdmg_b = st.slider("Crit DMG (B)", 1.2, 3.0, float(active_preset["b"]["cdmg"]) if active_preset else 1.8, step=0.1)

    with st.expander("✨ Status Effects & Buffs (Karakter B)"):
        buff_atk_b = st.slider("ATK Buff (%) [B]", 0, 100, int(active_preset["b"]["atk_buff"] * 100) if active_preset else 0, step=5) / 100.0
        debuff_def_b = st.slider("DEF Debuff (%) [B]", 0, 100, int(active_preset["b"]["def_debuff"] * 100) if active_preset else 0, step=5) / 100.0
        stun_b = st.checkbox("Mulai dalam keadaan STUN? [B]", value=active_preset["b"]["stun"] if active_preset else False)
        burn_b = st.number_input("DoT Burn Damage/turn (B)", min_value=0, max_value=500, value=int(active_preset["b"]["burn"]) if active_preset else 0, step=20)

st.divider()

# --- Object Construction ---
def create_characters():
    char_a = Character(
        name=name_a, max_hp=float(hp_a), current_hp=float(hp_a),
        atk=float(atk_a), defense=float(def_a), speed=float(spd_a),
        accuracy=float(acc_a), crit_rate=float(crit_a), crit_dmg=float(cdmg_a)
    )
    if buff_atk_a > 0:
        char_a.apply_effect(Effect(EffectType.ATK_BUFF, duration=3, value=buff_atk_a, proc_chance=1.0))
    if debuff_def_a > 0:
        char_a.apply_effect(Effect(EffectType.DEF_DEBUFF, duration=3, value=debuff_def_a, proc_chance=1.0))
    if stun_a:
        char_a.apply_effect(Effect(EffectType.STUN, duration=1, value=0.0, proc_chance=1.0))
    if burn_a > 0:
        char_a.apply_effect(Effect(EffectType.BURN, duration=3, value=float(burn_a), proc_chance=1.0))

    char_b = Character(
        name=name_b, max_hp=float(hp_b), current_hp=float(hp_b),
        atk=float(atk_b), defense=float(def_b), speed=float(spd_b),
        accuracy=float(acc_b), crit_rate=float(crit_b), crit_dmg=float(cdmg_b)
    )
    if buff_atk_b > 0:
        char_b.apply_effect(Effect(EffectType.ATK_BUFF, duration=3, value=buff_atk_b, proc_chance=1.0))
    if debuff_def_b > 0:
        char_b.apply_effect(Effect(EffectType.DEF_DEBUFF, duration=3, value=debuff_def_b, proc_chance=1.0))
    if stun_b:
        char_b.apply_effect(Effect(EffectType.STUN, duration=1, value=0.0, proc_chance=1.0))
    if burn_b > 0:
        char_b.apply_effect(Effect(EffectType.BURN, duration=3, value=float(burn_b), proc_chance=1.0))

    return char_a, char_b

char_a_obj, char_b_obj = create_characters()
features_dict = compute_features(char_a_obj, char_b_obj)
df_features = pd.DataFrame([features_dict])

# --- Prediction & Benchmark Action ---
col_action1, col_action2 = st.columns([2, 1])

with col_action1:
    btn_predict = st.button("⚡ Prediksi Hasil Pertarungan (ML Real-time)", type="primary", use_container_width=True)

with col_action2:
    run_benchmark = st.checkbox("🔬 Bandingkan dengan Monte Carlo Simulator (Ground Truth)", value=True)

if btn_predict or True:
    if not models_loaded:
        st.warning("Model belum dapat diakses.")
    else:
        # 1. ML Real-time Inference
        t_start_ml = time.perf_counter()
        pred_damage_a = reg_model.predict(df_features)[0]
        pred_win_prob_a = clf_model.predict_proba(df_features)[0][1]
        t_end_ml = time.perf_counter()
        ml_latency_ms = (t_end_ml - t_start_ml) * 1000

        pred_win_prob_b = 1.0 - pred_win_prob_a

        st.subheader("📊 Hasil Prediksi Machine Learning")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(
                label=f"Win Rate {name_a}",
                value=f"{pred_win_prob_a * 100:.1f}%",
                delta=f"{(pred_win_prob_a - 0.5) * 100:+.1f}% vs 50:50"
            )
        with m_col2:
            st.metric(
                label=f"Win Rate {name_b}",
                value=f"{pred_win_prob_b * 100:.1f}%",
                delta=f"{(pred_win_prob_b - 0.5) * 100:+.1f}% vs 50:50"
            )
        with m_col3:
            st.metric(
                label=f"Est. Total Dmg Dealt ({name_a})",
                value=f"{pred_damage_a:.1f}",
                delta=f"{pred_damage_a - hp_b:+.1f} vs Target HP"
            )
        with m_col4:
            st.metric(
                label="⚡ Inferensi Latency",
                value=f"{ml_latency_ms:.2f} ms",
                delta="Ultra Fast (<1ms)"
            )

        # Win Rate Progress Bar
        st.markdown(f"**Peluang Dominasi Pertarungan: {name_a} vs {name_b}**")
        st.progress(float(min(max(pred_win_prob_a, 0.0), 1.0)))

        # 2. Monte Carlo Comparison (jika dicentang)
        if run_benchmark:
            with st.spinner("Menjalankan 200 simulasi turn-based Monte Carlo untuk validasi akurasi..."):
                t_start_mc = time.perf_counter()
                mc_res = run_monte_carlo_duel(char_a_obj, char_b_obj, n_simulations=200)
                t_end_mc = time.perf_counter()
                mc_latency_ms = (t_end_mc - t_start_mc) * 1000

            st.markdown("#### ⚖️ Perbandingan: Machine Learning vs Monte Carlo Asli")
            comp_df = pd.DataFrame({
                "Metrik": [
                    f"Win Rate {name_a} (%)",
                    f"Win Rate {name_b} (%)",
                    f"Avg Total Damage {name_a}",
                    "Waktu Eksekusi (Latency)"
                ],
                "Prediksi ML (LightGBM/XGBoost)": [
                    f"{pred_win_prob_a * 100:.2f}%",
                    f"{pred_win_prob_b * 100:.2f}%",
                    f"{pred_damage_a:.2f}",
                    f"{ml_latency_ms:.2f} ms"
                ],
                "Simulasi Monte Carlo (200 Duel)": [
                    f"{mc_res['win_rate_a']:.2f}%",
                    f"{mc_res['win_rate_b']:.2f}%",
                    f"{mc_res['average_total_damage_a']:.2f}",
                    f"{mc_latency_ms:.2f} ms"
                ],
                "Akurasi / Selisih": [
                    f"Selisih: {abs(pred_win_prob_a * 100 - mc_res['win_rate_a']):.2f}%",
                    f"Selisih: {abs(pred_win_prob_b * 100 - mc_res['win_rate_b']):.2f}%",
                    f"Selisih: {abs(pred_damage_a - mc_res['average_total_damage_a']):.2f}",
                    f"🚀 {mc_latency_ms / max(0.001, ml_latency_ms):.0f}x Lebih Cepat!"
                ]
            })
            st.table(comp_df)

        # 3. Engineered Features Inspector
        with st.expander("🔍 Lihat Rekayasa Fitur (Feature Engineering) yang Dibaca Model"):
            st.write("Model tidak hanya membaca stat mentah, tetapi membaca relasi perbandingan antar kedua karakter:")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.write(f"• **HP Ratio (A/B)**: `{features_dict['hp_ratio']:.2f}`")
                st.write(f"• **ATK to DEF Ratio**: `{features_dict['atk_to_def_ratio']:.2f}`")
            with f_col2:
                st.write(f"• **Speed Advantage**: `{features_dict['speed_advantage']:+.1f}`")
                st.write(f"• **Stun Advantage**: `{features_dict['has_stun_advantage']}`")
            with f_col3:
                st.write(f"• **Effective DPS (A)**: `{features_dict['effective_dps_a']:.1f}`")
                st.write(f"• **Effective DPS (B)**: `{features_dict['effective_dps_b']:.1f}`")
                st.write(f"• **DoT Burn Potential**: `{features_dict['dot_burn_potential']:+.1f}`")
