import pandas as pd, json, numpy as np, os

P = "./"  # folder with your .parquet files

def read(name):
    return pd.read_parquet(os.path.join(P, name))

# Overall performance (players_summary has humans; use bot_skill_profiles if you prefer)
parsed = read("parsed_rows.parquet")
players = read("players_summary.parquet")

# Bot names present
bots = parsed.loc[parsed['is_bot']==True, 'player_name'].dropna().unique().tolist()
BOT = bots[0]  # choose one or loop

# MP% and Raw means for bot
bot_rows = parsed[parsed['player_name']==BOT]
metrics = {}
metrics['BOT_NAME'] = BOT
metrics['BOT_MP_MEAN'] = float(bot_rows['mp_pct'].mean())
metrics['BOT_RAW_MEAN'] = float(bot_rows['rawscore'].mean())

# Expert human baselines (if SkillBucket present in players)
if 'SkillBucket' in players.columns:
    experts = players[players['SkillBucket']=='Expert']
    metrics['EXP_MP_MEAN'] = float(experts['mean_mp'].mean())
    metrics['EXP_RAW_MEAN'] = float(experts['mean_raw'].mean())

# Bid match (from shadow or recompute via decision_key)
try:
    shx = read("shadow_analysis_expert_corrected.parquet")
    # Expect columns like: player_name, decision_key, match_expert (0/1)
    shb = shx[shx['player_name']==BOT]
    metrics['BID_MATCH_%'] = 100*float(shb['match_expert'].mean())
    metrics['N_BIDS'] = int(shb.shape[0])
except Exception:
    metrics['BID_MATCH_%'] = None; metrics['N_BIDS'] = None

# Opening lead consistency
try:
    ol = read("bot_opening_lead_match.parquet")
    ob = ol[ol['player_name']==BOT]
    metrics['LEAD_MATCH_%'] = 100*float(ob['lead_match_rate'].mean())
    metrics['N_LEADS'] = int(ob['n_leads'].sum() if 'n_leads' in ob.columns else parsed[(parsed['player_name']==BOT) & parsed['opening_lead'].notna()].shape[0])
except Exception:
    metrics['LEAD_MATCH_%'] = None; metrics['N_LEADS'] = None

# Contract distance
try:
    cd = read("bot_contract_distance.parquet")
    cb = cd[cd['player_name']==BOT]
    metrics['MEAN_CD'] = float(cb['mean_contract_distance'].mean())
except Exception:
    metrics['MEAN_CD'] = None

# Strategic metrics
for f, key in [("strategic_rates.parquet","game_rate"), ("strategic_rates.parquet","slam_rate")]:
    try:
        s = read(f); b = s[s['player_name']==BOT]
        metrics[key.upper()] = 100*float(b[key].mean())
    except Exception:
        metrics[key.upper()] = None

try:
    d = read("double_rate.parquet"); b = d[d['player_name']==BOT]
    metrics['DOUBLE_%'] = 100*float(b['double_rate'].mean())
except Exception:
    metrics['DOUBLE_%'] = None

try:
    ps = read("play_phase_sharpness.parquet"); b = ps[ps['player_name']==BOT]
    metrics['AVG_OVER'] = float(b['avg_overtricks'].mean())
    metrics['AVG_UNDER'] = float(b['avg_undertricks'].mean())
except Exception:
    metrics['AVG_OVER'] = metrics['AVG_UNDER'] = None

print(json.dumps(metrics, indent=2))
