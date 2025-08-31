# =========================
# BRIDGE PIPELINE for (louis, detail, registrants) files
# =========================
import os, re, json, hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# ------------- Paths -------------
# Use merged files instead of single date files
INPUT_DETAIL = "detail_20250701_20250710_merged.tsv"
INPUT_REG    = "registrants_20250701_20250710_merged.tsv"

# Bot files - find all merged bot files
BOT_FILES = [
    "louis_20250701_20250710_argine_merged",
    "louis_20250701_20250710_ben_advanced_merged", 
    "louis_20250701_20250710_ben_beginner_merged",
    "louis_20250701_20250710_ben_intermediate_merged",
    "louis_20250701_20250710_ben_novice_merged",
    "louis_20250701_20250710_gib_advanced_merged",
    "louis_20250701_20250710_gib_basic_merged"
]

OUT_DIR = "bridge_output_run"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- Bridge helpers -------------
SEATS = ["S","W","N","E"]
DEALER_CODE_TO_SEAT = {"1":"S","2":"W","3":"N","4":"E"}
RANKS = "23456789TJQKA"
RANK_TO_VAL = {r:i for i,r in enumerate(RANKS, start=2)}

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return ""

def parse_lin_tokens(movie: str) -> List[Tuple[str,str]]:
    if not isinstance(movie, str): return []
    parts = movie.split("|")
    toks = []
    for i in range(0, len(parts)-1, 2):
        t = parts[i].strip().lower()
        v = parts[i+1]
        if t:
            toks.append((t, v))
    return toks

def vulnerability_from_sv(val: str) -> Tuple[bool,bool]:
    v = safe_str(val).strip().lower()
    if v in {"o","0","n","none","-",""}: return (False, False)
    if v in {"b","both","a","all"}:       return (True, True)
    if v in {"ns","n-s","n/s"}:           return (True, False)
    if v in {"ew","e-w","e/w"}:           return (False, True)
    return (False, False)

def seat_cycle_from_dealer(dealer: Optional[str]) -> List[str]:
    dealer = dealer if dealer in SEATS else "S"
    i = SEATS.index(dealer)
    return SEATS[i:] + SEATS[:i]

@dataclass
class Contract:
    level: Optional[int] = None
    strain: Optional[str] = None  # S,H,D,C,N
    doubled: int = 0              # 0 none, 1 X, 2 XX
    declarer: Optional[str] = None

def clean_call(call_raw: str) -> str:
    if call_raw is None: return ""
    c = safe_str(call_raw).strip().upper().replace("NT","N")
    c = re.sub(r"[^A-Z0-9]", "", c)
    if c in {"P","X","XX"}:
        return c
    return c if re.fullmatch(r"[1-7][SHDCN]", c) else c

def parse_bidding_and_contract(tokens: List[Tuple[str,str]], dealer_seat: Optional[str]) -> Tuple[List[Tuple[str,str]], Contract]:
    auction: List[Tuple[str,str]] = []
    cycle = seat_cycle_from_dealer(dealer_seat)
    idx = 0
    for tag, val in tokens:
        if tag == "mb":
            call = clean_call(val)
            seat = cycle[idx % 4]
            auction.append((seat, call))
            idx += 1

    last_bid_i = None
    for i in range(len(auction)-1, -1, -1):
        if re.fullmatch(r"[1-7][SHDCN]", auction[i][1]):
            last_bid_i = i; break

    c = Contract()
    if last_bid_i is not None:
        last_call = auction[last_bid_i][1]
        c.level = int(last_call[0]); c.strain = last_call[1]
        dbl = 0
        for j in range(last_bid_i+1, len(auction)):
            cc = auction[j][1]
            if cc == "XX": dbl = 2
            elif cc == "X": dbl = max(dbl,1)
            elif cc == "P": continue
            elif re.fullmatch(r"[1-7][SHDCN]", cc):
                last_bid_i = j; c.level = int(cc[0]); c.strain = cc[1]; dbl = 0
        c.doubled = dbl

        # declarer = first from declaring side to bid the final strain
        last_bidder = auction[last_bid_i][0]
        side = {"N","S"} if last_bidder in {"N","S"} else {"E","W"}
        for s, cc in auction:
            if re.fullmatch(r"[1-7][SHDCN]", cc) and cc[1] == c.strain and s in side:
                c.declarer = s; break
        if c.declarer is None:
            c.declarer = last_bidder
    return auction, c

def parse_play(tokens: List[Tuple[str,str]]) -> List[str]:
    plays = []
    for tag, val in tokens:
        if tag == "pc":
            v = safe_str(val).strip().upper()
            m = re.match(r"^([SHDC])([2-9TJQKA])", v)
            if m:
                plays.append(m.group(1)+m.group(2))
    return plays

def trick_winner(cards: List[str], trump: Optional[str]) -> int:
    lead = cards[0][0]
    best = 0
    for i, c in enumerate(cards):
        s, r = c[0], c[1]
        if trump and trump in "SHDC":
            if cards[best][0] != trump:
                if s == trump: best = i
                elif s == lead and cards[best][0] == lead and RANK_TO_VAL[r] > RANK_TO_VAL[cards[best][1]]: best = i
            else:
                if s == trump and RANK_TO_VAL[r] > RANK_TO_VAL[cards[best][1]]: best = i
        else:
            if s == lead and RANK_TO_VAL[r] > RANK_TO_VAL[cards[best][1]]: best = i
    return best

def compute_tricks_made_from_plays(plays: List[str], declarer: Optional[str], strain: Optional[str]) -> Optional[int]:
    if not plays or not declarer or not strain: return None
    trump = None if strain == "N" else strain
    lead_idx = (SEATS.index(declarer) + 1) % 4
    i = 0; tm = 0
    while i < len(plays):
        trick = plays[i:i+4]
        if len(trick) < 4: break
        wrel = trick_winner(trick, trump)
        winner = SEATS[(lead_idx + wrel) % 4]
        if (winner in {"N","S"}) == (declarer in {"N","S"}): tm += 1
        lead_idx = (lead_idx + wrel) % 4
        i += 4
    return tm

def md_to_hcp(md_val: str) -> Dict[str, Optional[int]]:
    # Count HCP (A=4,K=3,Q=2,J=1), accepts lower/upper case honors
    out = {s: None for s in SEATS}
    if not isinstance(md_val, str) or not md_val.strip():
        return out
    s = md_val.strip()
    rest = s[1:] if s[:1] in "1234" else s
    hands = re.split(r"[ ,;]+", rest.strip())
    if len(hands) != 4:
        return out
    order = ["S","W","N","E"]
    for seat, hand in zip(order, hands):
        suits = (hand.split(".") + ["","","",""])[:4]
        h = 0
        for suit in suits:
            for r in suit:
                if r.upper() == "A": h += 4
                elif r.upper() == "K": h += 3
                elif r.upper() == "Q": h += 2
                elif r.upper() == "J": h += 1
        out[seat] = h
    return out

def compute_raw_score(contract: Contract, tricks_made: Optional[int], ns_vul: bool, ew_vul: bool) -> Optional[int]:
    if contract is None or contract.level is None or contract.strain is None or contract.declarer is None or tricks_made is None:
        return None
    declarer_ns = contract.declarer in {"N","S"}
    vul = ns_vul if declarer_ns else ew_vul
    target = 6 + contract.level
    made = tricks_made - target
    score = 0
    d = contract.doubled
    if made >= 0:
        if contract.strain == "N":
            base = 40 + (contract.level - 1) * 30
        elif contract.strain in {"H","S"}:
            base = 30 * contract.level
        else:
            base = 20 * contract.level
        if d == 1: base *= 2
        if d == 2: base *= 4
        score += base
        if d == 1: score += 50
        if d == 2: score += 100
        game = base >= 100
        if contract.level == 6:
            score += 1250 if vul else 750
        elif contract.level == 7:
            score += 2250 if vul else 1500
        else:
            score += 500 if (vul and game) else (300 if game else 50)
        if made > 0:
            if d == 0:
                per = 30 if contract.strain in {"N","H","S"} else 20
                score += per * made
            else:
                per = (200 if vul else 100) * (2 if d == 2 else 1)
                score += per * made
    else:
        down = -made
        if d == 0:
            score -= (100 if vul else 50) * down
        else:
            if vul:
                penalties = [200] + [300]*(down-1)
            else:
                penalties = [100, 200, 200] + [300]*max(0, down-3)
            pen = sum(penalties[:down])
            if d == 2: pen *= 2
            score -= pen
    return score

# ------------- Ingest: LOUIS (bot) file -------------
def parse_multiple_louis_files(file_paths: List[str]) -> pd.DataFrame:
    """Parse multiple louis bot files and combine them into a single DataFrame."""
    all_bot_data = []
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found, skipping...")
            continue
            
        print(f"Processing bot file: {path}")
        bot_df = parse_louis_file(path)
        all_bot_data.append(bot_df)
    
    if not all_bot_data:
        raise ValueError("No valid bot files found!")
        
    # Combine all bot dataframes
    combined_bot_df = pd.concat(all_bot_data, ignore_index=True, sort=False)
    print(f"Combined {len(all_bot_data)} bot files into {len(combined_bot_df)} rows")
    
    return combined_bot_df

def parse_louis_file(path: str) -> pd.DataFrame:
    # Extract bot name from merged file names like "louis_20250701_20250710_ben_advanced_merged.tsv"
    basename = os.path.basename(path)
    if "_merged.tsv" in basename:
        # For merged files, extract the bot type from the filename
        # louis_20250701_20250710_ben_advanced_merged.tsv -> ben_advanced
        parts = basename.replace("_merged.tsv", "").split("_")
        if len(parts) >= 4:  # louis_20250701_20250710_bottype_level
            bot_name = "_".join(parts[3:])  # Everything after the date range
        else:
            bot_name = parts[-1] if parts else basename
    else:
        # Original format: louis.20250701.ben.advanced -> ben.advanced
        bot_name = os.path.basename(path).split(".", 2)[-1]
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("RESULT"):
            i += 1; continue
        # RESULT line
        kv = {}
        for tok in line.split():
            if tok == "RESULT": continue
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k.strip()] = v.strip()
        # find next MOVIE
        movie = None
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if nxt.startswith("MOVIE"):
                movie = nxt[len("MOVIE"):].strip()
                break
            elif nxt.startswith("RESULT"):
                break
            j += 1
        rows.append({
            "tourney_id": kv.get("tourney"),
            "board_number": pd.to_numeric(kv.get("board"), errors="coerce"),
            "instance_id": pd.to_numeric(kv.get("instance"), errors="coerce"),
            "result_str": kv.get("result"),
            "rawscore": pd.to_numeric(kv.get("rawscore"), errors="coerce"),
            "mp_pct_reported": pd.to_numeric(kv.get("mp%"), errors="coerce"),
            "movie": movie,
            "player_name": bot_name,
            "is_bot": True
        })
        i = j + 1 if j > i else i + 1
    return pd.DataFrame(rows)

# ------------- Ingest: human detail + registrants -------------
def load_humans(detail_path: str, reg_path: str) -> pd.DataFrame:
    det = pd.read_csv(detail_path, sep="\t", dtype=str)
    lc = {c.lower(): c for c in det.columns}
    def pick(*cands):
        for c in cands:
            if c in lc: return lc[c]
        return None
    tcol = pick("tourney_id","tourney")
    team_col = pick("team_id","team")
    bcol = pick("board_number","board")
    icol = pick("instance_id","instance")
    rcol = pick("result")
    scol = pick("rawscore","score")
    mcol = pick("final_score_movie","movie","lin")
    df = det[[tcol,team_col,bcol,icol,rcol,scol,mcol]].copy()
    df.columns = ["tourney_id","team_id","board_number","instance_id","result_str","rawscore","movie"]
    df["board_number"] = pd.to_numeric(df["board_number"], errors="coerce")
    df["instance_id"] = pd.to_numeric(df["instance_id"], errors="coerce")
    df["rawscore"] = pd.to_numeric(df["rawscore"], errors="coerce")
    df["is_bot"] = False

    reg = pd.read_csv(reg_path, sep="\t", dtype=str)
    rl = {c.lower(): c for c in reg.columns}
    team_r = next((rl[c] for c in ["team_id","team","teamid"] if c in rl), None)
    name_r = next((rl[c] for c in ["player_name","name","team_name","display_name","username"] if c in rl), None)
    if team_r and name_r:
        df = df.merge(reg[[team_r, name_r]].drop_duplicates().rename(columns={team_r:"team_id", name_r:"player_name"}),
                      on="team_id", how="left")
    else:
        df["player_name"] = df["team_id"]
    return df

# ------------- LIN feature extraction -------------
def extract_lin(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, r in df.iterrows():
        mv = r.get("movie", None)
        toks = parse_lin_tokens(mv) if isinstance(mv, str) else []
        tags = defaultdict(list)
        for tag, val in toks:
            tags[tag].append(val)
        md_raw = tags["md"][0] if tags["md"] else None
        dealer = DEALER_CODE_TO_SEAT.get(md_raw[0], None) if isinstance(md_raw, str) and md_raw[:1].isdigit() else None
        ns_v, ew_v = vulnerability_from_sv(tags["sv"][0] if tags["sv"] else "")
        auction, contract = parse_bidding_and_contract(toks, dealer)
        plays = parse_play(toks)
        opening_lead = plays[0] if plays else None
        tricks = compute_tricks_made_from_plays(plays, contract.declarer, contract.strain) if plays and contract.declarer and contract.strain else None
        hcp = md_to_hcp(md_raw or "")
        rows.append({
            "idx": idx,
            "md_raw": md_raw,
            "dealer": dealer,
            "ns_vulnerable": ns_v,
            "ew_vulnerable": ew_v,
            "final_level": contract.level,
            "final_strain": contract.strain,
            "final_doubled": contract.doubled,
            "declarer": contract.declarer,
            "opening_lead": opening_lead,
            "tricks_made": tricks,
            "S_HCP": hcp["S"], "W_HCP": hcp["W"], "N_HCP": hcp["N"], "E_HCP": hcp["E"]
        })
    feat = pd.DataFrame(rows).set_index("idx")
    out = df.join(feat, how="left")
    # create deal key
    out["tourney_id"] = out["tourney_id"].astype(str)
    out["deal_key"] = out["tourney_id"] + "::" + out["board_number"].astype("Int64").astype(str) + "::" + out["instance_id"].astype("Int64").astype(str)
    # prefer explicit rawscore if present
    return out

# ------------- Bidding actions table -------------
def build_bidding_actions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        mv = row.get("movie", None)
        if not isinstance(mv, str) or not mv.strip():
            continue
        toks = parse_lin_tokens(mv)
        md_raw = next((v for t, v in toks if t == "md"), None)
        dealer = DEALER_CODE_TO_SEAT.get(md_raw[0], None) if isinstance(md_raw, str) and md_raw[:1].isdigit() else None
        auction, _ = parse_bidding_and_contract(toks, dealer)
        prefix = []
        for k, (seat, call) in enumerate(auction):
            decision_key = f"{row['deal_key']}::BID::{seat}::" + ",".join(prefix)
            rows.append({
                "deal_key": row["deal_key"],
                "tourney_id": row["tourney_id"],
                "board_number": row["board_number"],
                "instance_id": row["instance_id"],
                "player_name": row.get("player_name", None),
                "is_bot": bool(row.get("is_bot", False)),
                "seat": seat,
                "call": call,
                "decision_index": k,
                "decision_key": decision_key
            })
            prefix.append(f"{seat}:{call}")
    return pd.DataFrame(rows)

# ------------- Matchpoint % per deal (tie-aware) -------------
def mp_pct_tieaware(scores: pd.Series) -> pd.Series:
    n = scores.shape[0]
    if n <= 1:
        return pd.Series([np.nan]*n, index=scores.index)
    vals = scores.values
    out = []
    for i, s in enumerate(vals):
        if pd.isna(s): out.append(np.nan); continue
        beats = np.sum(vals < s)
        ties = np.sum(vals == s) - 1
        out.append(((beats + 0.5*ties)/(n-1))*100.0)
    return pd.Series(out, index=scores.index)

# ------------- Contract category -------------
def contract_category(level, strain):
    if pd.isna(level) or pd.isna(strain): return "Unknown"
    lvl = int(level); st = str(strain)
    if lvl == 7: return "Grand Slam"
    if lvl == 6: return "Small Slam"
    if (st == "N" and lvl >= 3) or (st in {"H","S"} and lvl >= 4) or (st in {"C","D"} and lvl >= 5): return "Game"
    if 1 <= lvl <= 5: return "Part-score"
    return "Unknown"

# ------------- Main -------------
# Ingest
bot_raw = parse_multiple_louis_files(BOT_FILES)
hum_raw = load_humans(INPUT_DETAIL, INPUT_REG)

# Parse LIN
bot = extract_lin(bot_raw)
hum = extract_lin(hum_raw)

# Combine
all_rows = pd.concat([hum, bot], ignore_index=True, sort=False)

# Local MP% per deal_key (for humans who don't have reported MP%)
all_rows["mp_pct_local"] = all_rows.groupby("deal_key")["rawscore"].transform(mp_pct_tieaware)
# Use reported mp% for bots (their actual tournament performance); use local mp% for humans
all_rows["mp_pct"] = np.where(all_rows["is_bot"] == True, all_rows["mp_pct_reported"], all_rows["mp_pct_local"])

# Save parsed rows with proper Parquet formatting for better performance and data type preservation
# Drop tuple columns that can't be serialized to Parquet, keep string representations
all_rows_for_save = all_rows.copy()
if 'contract_tuple' in all_rows_for_save.columns:
    all_rows_for_save = all_rows_for_save.drop(['contract_tuple'], axis=1)

parsed_rows_path = os.path.join(OUT_DIR, "parsed_rows.parquet")
all_rows_for_save.to_parquet(parsed_rows_path, index=False)

# Bidding actions
bids = build_bidding_actions(all_rows)
bids_path = os.path.join(OUT_DIR, "bidding_actions.parquet")
bids.to_parquet(bids_path, index=False)

# Opening lead suit
all_rows["lead_suit"] = all_rows["opening_lead"].astype(str).str[0]

# ========== Human Skill Bucketing ==========
hum_only = all_rows[all_rows["is_bot"] == False].copy()
hum_agg = hum_only.groupby("player_name").agg(
    mean_mp=("mp_pct","mean"),
    mean_raw=("rawscore","mean"),
    boards=("deal_key","count")
).reset_index()

def minmax(series):
    x = series.astype(float)
    lo = np.nanmin(x) if x.notna().any() else np.nan
    hi = np.nanmax(x) if x.notna().any() else np.nan
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.full(len(x), np.nan), index=series.index)
    return (x - lo) / (hi - lo) * 100.0

hum_agg["mp_scaled"] = minmax(hum_agg["mean_mp"])
hum_agg["raw_scaled"] = minmax(hum_agg["mean_raw"])

# Handle empty arrays safely
mp_scaled = hum_agg["mp_scaled"].fillna(0).values
raw_scaled = hum_agg["raw_scaled"].fillna(0).values

if len(mp_scaled) > 0 and len(raw_scaled) > 0:
    stacked = np.vstack([mp_scaled, raw_scaled])
    # Only compute mean where both values are not NaN
    with np.errstate(invalid='ignore'):
        hum_agg["composite"] = np.nanmean(stacked, axis=0)
    # Replace zeros back to NaN where original data was NaN
    mask = hum_agg["mp_scaled"].isna() & hum_agg["raw_scaled"].isna()
    hum_agg.loc[mask, "composite"] = np.nan
else:
    hum_agg["composite"] = np.full(len(hum_agg), np.nan)

vals = hum_agg["composite"].dropna().values
if len(vals) >= 20:
    qs = np.percentile(vals, [0,20,50,80,95,100])
    def to_bucket(x):
        if np.isnan(x): return "Unknown"
        if x <= qs[1]: return "Beginner"
        if x <= qs[2]: return "Novice"
        if x <= qs[3]: return "Intermediate"
        if x <= qs[4]: return "Advanced"
        return "Expert"
else:
    def to_bucket(x):
        if np.isnan(x): return "Unknown"
        if x <= 20: return "Beginner"
        if x <= 40: return "Novice"
        if x <= 60: return "Intermediate"
        if x <= 80: return "Advanced"
        return "Expert"
hum_agg["SkillBucket"] = hum_agg["composite"].apply(to_bucket)

players_summary_path = os.path.join(OUT_DIR, "players_summary.parquet")
hum_agg.to_parquet(players_summary_path, index=False)

# Merge skill onto rows & bids
all_rows = all_rows.merge(hum_agg[["player_name","SkillBucket"]], on="player_name", how="left")
bids = bids.merge(hum_agg[["player_name","SkillBucket"]], on="player_name", how="left")

# ========== Metrics ==========

# Contract category, game/slam flags
all_rows["contract_category"] = [contract_category(lv, st) for lv, st in zip(all_rows["final_level"], all_rows["final_strain"])]
all_rows["is_game"] = all_rows["contract_category"] == "Game"
all_rows["is_slam"] = all_rows["contract_category"].isin(["Small Slam","Grand Slam"])

# Over/undertricks
def over_under_row(r):
    if pd.isna(r["tricks_made"]) or pd.isna(r["final_level"]): return pd.Series([np.nan, np.nan])
    target = 6 + int(r["final_level"])
    delta = int(r["tricks_made"]) - target
    return pd.Series([max(delta,0), max(-delta,0)])
ou = all_rows.apply(over_under_row, axis=1)
all_rows["overtricks"] = ou[0]; all_rows["undertricks"] = ou[1]

# Performance by contract category & strain
perf_by_cat_strain = (all_rows.groupby(["is_bot","player_name","contract_category","final_strain"])
                      .agg(mean_rawscore=("rawscore","mean"),
                           boards=("deal_key","count"))
                      .reset_index())
perf_by_cat_strain.to_parquet(os.path.join(OUT_DIR, "performance_by_contract_and_strain.parquet"), index=False)

# Decision similarity (bidding) vs expert human mode action per decision_key
expert_bids = bids[(bids["is_bot"] == False) & (bids["SkillBucket"] == "Expert")]
mode_by_key = expert_bids.groupby("decision_key")["call"].agg(lambda x: Counter(x).most_common(1)[0][0] if len(x) else np.nan).rename("expert_mode_call")
bids_eval = bids.merge(mode_by_key, on="decision_key", how="left")
bids_eval["match_expert"] = (bids_eval["call"] == bids_eval["expert_mode_call"]).astype(float)
(bids_eval[bids_eval["is_bot"] == True]
 .groupby("player_name")
 .agg(bid_match_rate=("match_expert","mean"), n_decisions=("match_expert","count"))
 .reset_index()
 .to_parquet(os.path.join(OUT_DIR, "bot_bid_match.parquet"), index=False))

# Opening lead consistency vs expert human suit mode per deal
exp_lead_mode = (all_rows[(all_rows["is_bot"] == False) & (all_rows["SkillBucket"] == "Expert")]
                 .groupby("deal_key")["lead_suit"]
                 .agg(lambda x: Counter(x.dropna()).most_common(1)[0][0] if x.notna().any() else np.nan)
                 .rename("expert_lead_suit"))
lead_eval = all_rows.merge(exp_lead_mode, on="deal_key", how="left")
lead_eval["lead_match_expert"] = (lead_eval["lead_suit"] == lead_eval["expert_lead_suit"]).astype(float)
(lead_eval[lead_eval["is_bot"] == True]
 .groupby("player_name")
 .agg(lead_match_rate=("lead_match_expert","mean"), n_leads=("lead_match_expert","count"))
 .reset_index()
 .to_parquet(os.path.join(OUT_DIR, "bot_opening_lead_match.parquet"), index=False))

# Bidding vs play proxies (over/undertricks)
(all_rows.groupby(["is_bot","player_name"])
 .agg(avg_overtricks=("overtricks","mean"),
      avg_undertricks=("undertricks","mean"))
 .reset_index()
 .to_parquet(os.path.join(OUT_DIR, "play_phase_sharpness.parquet"), index=False))

# Doubling frequency (including both doubles "D" and redoubles "R")
bids["is_double"] = bids["call"].isin(["D","R"]).astype(int)
(bids.groupby(["is_bot","player_name"]).agg(double_rate=("is_double","mean")).reset_index()
 .to_parquet(os.path.join(OUT_DIR, "double_rate.parquet"), index=False))

# Strategic rates
(all_rows.groupby(["is_bot","player_name"])
 .agg(game_rate=("is_game","mean"), slam_rate=("is_slam","mean"))
 .reset_index()
 .to_parquet(os.path.join(OUT_DIR, "strategic_rates.parquet"), index=False))

# Thin contract indicator (HCP thresholds)
def thin_flag(r):
    if pd.isna(r["final_level"]) or pd.isna(r["final_strain"]) or pd.isna(r["declarer"]):
        return np.nan
    lvl = int(r["final_level"]); st = str(r["final_strain"]); dec = str(r["declarer"])
    ns_hcp = (r["N_HCP"] or 0) + (r["S_HCP"] or 0)
    ew_hcp = (r["E_HCP"] or 0) + (r["W_HCP"] or 0)
    phcp = ns_hcp if dec in {"N","S"} else ew_hcp
    thresh = None
    if st == "N" and lvl >= 3: thresh = 25
    if st in {"H","S"} and lvl >= 4: thresh = 25
    if st in {"C","D"} and lvl >= 5: thresh = 29
    if lvl == 6: thresh = 33
    if lvl == 7: thresh = 37
    if thresh is None: return 0
    return 1 if phcp < thresh else 0

all_rows["thin_contract"] = all_rows.apply(thin_flag, axis=1)
(all_rows.groupby(["is_bot","player_name"]).agg(thin_rate=("thin_contract","mean")).reset_index()
 .to_parquet(os.path.join(OUT_DIR, "thin_contract_rate.parquet"), index=False))

# Human-likeness: contract decision distance vs expert
def to_contract_tuple(lv, st):
    if pd.isna(lv) or pd.isna(st): return None
    return (int(lv), str(st))

def to_contract_string(lv, st):
    if pd.isna(lv) or pd.isna(st): return None
    return f"{int(lv)}{str(st)}"

all_rows["contract_tuple"] = [to_contract_tuple(lv, st) for lv, st in zip(all_rows["final_level"], all_rows["final_strain"])]
all_rows["contract_string"] = [to_contract_string(lv, st) for lv, st in zip(all_rows["final_level"], all_rows["final_strain"])]

expert_contract_by_deal = (all_rows[(all_rows["is_bot"] == False) & (all_rows["SkillBucket"] == "Expert")]
                           .groupby("deal_key")["contract_string"]
                           .agg(lambda x: Counter([y for y in x if y is not None]).most_common(1)[0][0] if any(y is not None for y in x) else None)
                           .rename("expert_contract_string"))
d3 = all_rows.merge(expert_contract_by_deal, on="deal_key", how="left")

def contract_distance_from_strings(a, b):
    if a is None or b is None: return np.nan
    if pd.isna(a) or pd.isna(b): return np.nan
    if not isinstance(a, str) or not isinstance(b, str): return np.nan
    if len(a) < 2 or len(b) < 2: return np.nan
    
    # Parse level and strain from strings like "3N", "4S"
    try:
        l1, s1 = int(a[0]), a[1]
        l2, s2 = int(b[0]), b[1]
        d = abs(l1 - l2)
        if s1 != s2: d += 1
        return d
    except (ValueError, IndexError):
        return np.nan

d3["contract_distance_vs_expert"] = [contract_distance_from_strings(a, b) for a, b in zip(d3["contract_string"], d3["expert_contract_string"])]
(d3[d3["is_bot"] == True].groupby("player_name")
 .agg(mean_contract_distance=("contract_distance_vs_expert","mean"), boards=("deal_key","count"))
 .reset_index()
 .to_parquet(os.path.join(OUT_DIR, "bot_contract_distance.parquet"), index=False))

# Outliers
human_median = all_rows[all_rows["is_bot"] == False].groupby("deal_key")["rawscore"].median().rename("human_median_rawscore")
d4 = all_rows.merge(human_median, on="deal_key", how="left")
d4["score_diff_vs_humans"] = d4["rawscore"] - d4["human_median_rawscore"]
perf_outliers = d4[(d4["is_bot"] == True) & (d4["score_diff_vs_humans"].abs() >= 500)]

mismatch_counts = (bids_eval[bids_eval["is_bot"] == True]
                   .assign(unmatched=lambda x: (x["match_expert"] == 0).astype(int))
                   .groupby(["deal_key","player_name"])["unmatched"].sum().reset_index())

human_contract_set = (all_rows[all_rows["is_bot"] == False]
                      .groupby("deal_key")["contract_string"].apply(lambda x: set([y for y in x if y is not None]))
                      .rename("human_contract_set"))
cd = d3[d3["is_bot"] == True][["deal_key","player_name","contract_string"]].merge(human_contract_set, on="deal_key", how="left")
cd["contract_discrepancy"] = [ct not in (s if isinstance(s, set) else set()) for ct, s in zip(cd["contract_string"], cd["human_contract_set"])]

# For Parquet compatibility, convert the sets to strings for storage
cd["human_contracts_list"] = [','.join(sorted(s)) if isinstance(s, set) and s else '' for s in cd["human_contract_set"]]

outliers = (perf_outliers[["deal_key","player_name","rawscore","human_median_rawscore","score_diff_vs_humans"]]
            .merge(mismatch_counts, on=["deal_key","player_name"], how="outer")
            .merge(cd[["deal_key","player_name","contract_string","contract_discrepancy","human_contracts_list"]], on=["deal_key","player_name"], how="outer"))
outliers.to_parquet(os.path.join(OUT_DIR, "outliers.parquet"), index=False)

# ========== 1. Human Skill Bucket ID Cards ==========
print("Creating skill bucket analysis...")

# Comprehensive skill bucket analysis
skill_bucket_analysis = []
for bucket in hum_agg["SkillBucket"].unique():
    if bucket == "Unknown":
        continue
    
    bucket_players = hum_agg[hum_agg["SkillBucket"] == bucket]
    bucket_rows = all_rows[(all_rows["is_bot"] == False) & (all_rows["SkillBucket"] == bucket)]
    bucket_bids = bids[(bids["is_bot"] == False) & (bids["SkillBucket"] == bucket)]
    
    if len(bucket_rows) == 0:
        continue
    
    # Basic stats
    n_players = len(bucket_players)
    n_boards = bucket_rows["deal_key"].count()
    
    # Performance metrics
    avg_mp = bucket_rows["mp_pct"].mean()
    avg_raw = bucket_rows["rawscore"].mean()
    
    # Contract distribution
    contract_dist = bucket_rows["contract_category"].value_counts(normalize=True)
    part_score_rate = contract_dist.get("Part-score", 0)
    game_rate = contract_dist.get("Game", 0)
    small_slam_rate = contract_dist.get("Small Slam", 0)
    grand_slam_rate = contract_dist.get("Grand Slam", 0)
    
    # Strain preferences
    strain_dist = bucket_rows["final_strain"].value_counts(normalize=True)
    nt_rate = strain_dist.get("N", 0)
    major_rate = strain_dist.get("H", 0) + strain_dist.get("S", 0)
    minor_rate = strain_dist.get("C", 0) + strain_dist.get("D", 0)
    
    # Bidding behavior
    double_rate = bucket_bids["is_double"].mean() if len(bucket_bids) > 0 else 0
    
    # Playing performance  
    avg_overtricks = bucket_rows["overtricks"].mean()
    avg_undertricks = bucket_rows["undertricks"].mean()
    
    # Thin contracts
    thin_rate = bucket_rows["thin_contract"].mean()
    
    skill_bucket_analysis.append({
        "skill_bucket": bucket,
        "n_players": n_players,
        "n_boards": n_boards,
        "avg_mp_pct": avg_mp,
        "avg_raw_score": avg_raw,
        "part_score_rate": part_score_rate,
        "game_rate": game_rate,
        "small_slam_rate": small_slam_rate,
        "grand_slam_rate": grand_slam_rate,
        "nt_rate": nt_rate,
        "major_suit_rate": major_rate,
        "minor_suit_rate": minor_rate,
        "double_rate": double_rate,
        "avg_overtricks": avg_overtricks,
        "avg_undertricks": avg_undertricks,
        "thin_contract_rate": thin_rate
    })

skill_bucket_df = pd.DataFrame(skill_bucket_analysis)
skill_bucket_df.to_parquet(os.path.join(OUT_DIR, "skill_bucket_profiles.parquet"), index=False)

# Create a function to assign skill buckets to bots based on MP%
def assign_bot_skill_bucket(bot_mp_pct, skill_bucket_df):
    """Assign a skill bucket to a bot based on its MP% performance."""
    if skill_bucket_df.empty or pd.isna(bot_mp_pct):
        return "Unknown"
    
    # Find the skill bucket with the closest MP% to the bot's MP%
    mp_differences = abs(skill_bucket_df["avg_mp_pct"] - bot_mp_pct)
    closest_idx = mp_differences.idxmin()
    return skill_bucket_df.loc[closest_idx, "skill_bucket"]

# Create a mapping of bot names to their assigned skill buckets
bot_skill_mapping = {}
for bot_name in all_rows[all_rows["is_bot"] == True]["player_name"].unique():
    bot_rows = all_rows[all_rows["player_name"] == bot_name]
    if len(bot_rows) > 0:
        # Use calculated MP% for consistency (both bots and humans use this column after processing)
        bot_mp = bot_rows["mp_pct"].mean()
        assigned_bucket = assign_bot_skill_bucket(bot_mp, skill_bucket_df)
        bot_skill_mapping[bot_name] = assigned_bucket

print(f"Bot skill bucket assignments: {bot_skill_mapping}")

# ========== 2. Bot Skill Matching and ID Cards ==========
print("Creating bot skill matching analysis...")

bot_analysis = []
bot_names = all_rows[all_rows["is_bot"] == True]["player_name"].unique()

for bot_name in bot_names:
    bot_rows = all_rows[all_rows["player_name"] == bot_name]
    bot_bids = bids[bids["player_name"] == bot_name]
    
    if len(bot_rows) == 0:
        continue
        
    # Bot performance metrics
    bot_mp = bot_rows["mp_pct"].mean()
    bot_raw = bot_rows["rawscore"].mean()
    n_boards = len(bot_rows)
    
    # Contract distribution
    contract_dist = bot_rows["contract_category"].value_counts(normalize=True)
    part_score_rate = contract_dist.get("Part-score", 0)
    game_rate = contract_dist.get("Game", 0)
    small_slam_rate = contract_dist.get("Small Slam", 0)
    grand_slam_rate = contract_dist.get("Grand Slam", 0)
    
    # Strain preferences
    strain_dist = bot_rows["final_strain"].value_counts(normalize=True)
    nt_rate = strain_dist.get("N", 0)
    major_rate = strain_dist.get("H", 0) + strain_dist.get("S", 0)
    minor_rate = strain_dist.get("C", 0) + strain_dist.get("D", 0)
    
    # Bidding behavior
    double_rate = bot_bids["is_double"].mean() if len(bot_bids) > 0 else 0
    
    # Playing performance
    avg_overtricks = bot_rows["overtricks"].mean()
    avg_undertricks = bot_rows["undertricks"].mean()
    
    # Thin contracts
    thin_rate = bot_rows["thin_contract"].mean()
    
    # Find closest human skill bucket
    distances = []
    for _, bucket_row in skill_bucket_df.iterrows():
        # Calculate distance based on multiple metrics
        mp_diff = abs(bot_mp - bucket_row["avg_mp_pct"])
        raw_diff = abs(bot_raw - bucket_row["avg_raw_score"]) / 100  # Scale raw score difference
        game_diff = abs(game_rate - bucket_row["game_rate"]) * 100
        slam_diff = abs((small_slam_rate + grand_slam_rate) - (bucket_row["small_slam_rate"] + bucket_row["grand_slam_rate"])) * 100
        
        total_distance = mp_diff + raw_diff + game_diff + slam_diff
        distances.append((bucket_row["skill_bucket"], total_distance))
    
    closest_bucket = min(distances, key=lambda x: x[1])[0] if distances else "Unknown"
    closest_distance = min(distances, key=lambda x: x[1])[1] if distances else 0
    
    bot_analysis.append({
        "bot_name": bot_name,
        "n_boards": n_boards,
        "avg_mp_pct": bot_mp,
        "avg_raw_score": bot_raw,
        "closest_skill_bucket": closest_bucket,
        "distance_to_bucket": closest_distance,
        "part_score_rate": part_score_rate,
        "game_rate": game_rate,
        "small_slam_rate": small_slam_rate,
        "grand_slam_rate": grand_slam_rate,
        "nt_rate": nt_rate,
        "major_suit_rate": major_rate,
        "minor_suit_rate": minor_rate,
        "double_rate": double_rate,
        "avg_overtricks": avg_overtricks,
        "avg_undertricks": avg_undertricks,
        "thin_contract_rate": thin_rate
    })

bot_profiles_df = pd.DataFrame(bot_analysis)
bot_profiles_df.to_parquet(os.path.join(OUT_DIR, "bot_skill_profiles.parquet"), index=False)

# ========== 3. Risk Analysis - Comprehensive Doubling Behavior ==========
print("Creating comprehensive risk analysis...")

# Better approach: Analyze all doubling decisions, not just when opponents failed
risk_analysis = []

# Group by deal to analyze each bidding scenario
for deal_key in all_rows["deal_key"].unique():
    deal_data = all_rows[all_rows["deal_key"] == deal_key].copy()
    deal_bids = bids[bids["deal_key"] == deal_key].copy()
    
    if len(deal_data) == 0:
        continue
    
    # Get all doubles made in this deal
    doubles_made = deal_bids[deal_bids["is_double"] == 1]
    
    # Analyze each player's behavior in this deal
    for _, player_row in deal_data.iterrows():
        player_name = player_row["player_name"]
        
        # Did this player double in this deal?
        player_doubled = len(doubles_made[doubles_made["player_name"] == player_name]) > 0
        
        # Analyze the outcome - did doubling opportunities exist?
        other_players = deal_data[deal_data["player_name"] != player_name]
        any_failures = len(other_players[other_players["rawscore"] < 0]) > 0
        
        # For bots, use MP%-based skill bucket assignment; for humans use actual skill bucket
        if player_row["is_bot"]:
            skill_bucket = bot_skill_mapping.get(player_name, "Unknown")
        else:
            skill_bucket = player_row.get("SkillBucket", "Unknown")
        
        # Count this as a doubling opportunity if others failed
        if any_failures:
            risk_analysis.append({
                "deal_key": deal_key,
                "player_name": player_name,
                "is_bot": player_row["is_bot"],
                "skill_bucket": skill_bucket,
                "player_score": player_row["rawscore"],
                "others_failed": any_failures,
                "player_doubled": player_doubled,
                "contract_level": player_row["final_level"],
                "contract_strain": player_row["final_strain"],
                "thin_contract": player_row["thin_contract"],
                "player_failed": player_row["rawscore"] < 0
            })

risk_df = pd.DataFrame(risk_analysis)

# Enhanced risk metrics by player
if len(risk_df) > 0:
    # Calculate comprehensive risk metrics
    risk_summary = risk_df.groupby(["player_name", "is_bot", "skill_bucket"]).agg({
        "others_failed": "sum",        # Total doubling opportunities
        "player_doubled": "sum",       # Times they actually doubled
        "thin_contract": "mean",       # Thin contract rate
        "contract_level": "mean",      # Average contract level
        "player_failed": "mean"        # Failure rate
    }).reset_index()
    
    # Calculate risk metrics
    risk_summary["doubling_opportunity_rate"] = (
        risk_summary["player_doubled"] / risk_summary["others_failed"]
    ).fillna(0)
    
    risk_summary["risk_aversion"] = 1 - risk_summary["doubling_opportunity_rate"]
    
    # Add total deals for context
    total_deals = risk_df.groupby(["player_name", "is_bot", "skill_bucket"]).size().reset_index(name="total_deals")
    risk_summary = risk_summary.merge(total_deals, on=["player_name", "is_bot", "skill_bucket"])
    
    risk_summary.columns = [
        "player_name", "is_bot", "skill_bucket", "doubling_opportunities", 
        "times_doubled", "avg_thin_contract_rate", "avg_contract_level", "failure_rate",
        "doubling_opportunity_rate", "risk_aversion_score", "total_deals"
    ]
    
    risk_summary.to_parquet(os.path.join(OUT_DIR, "risk_analysis.parquet"), index=False)
    
    # Create separate risk analysis by player type for easier analysis
    bot_risk = risk_summary[risk_summary["is_bot"] == True].copy()
    human_risk = risk_summary[risk_summary["is_bot"] == False].copy()
    
    print(f"Risk analysis created: {len(risk_summary)} players")
    print(f"- Bots: {len(bot_risk)}")
    print(f"- Humans: {len(human_risk)}")
else:
    print("No risk analysis data generated")

# ========== 4. Contract Level Distribution Analysis ==========
print("Creating contract level distribution analysis...")

# Detailed contract analysis by player type and skill
contract_analysis = []

# Analyze humans by skill bucket
for bucket in skill_bucket_df["skill_bucket"]:
    bucket_data = all_rows[(all_rows["is_bot"] == False) & (all_rows["SkillBucket"] == bucket)]
    
    if len(bucket_data) == 0:
        continue
    
    # Contract level distribution
    level_dist = bucket_data["final_level"].value_counts(normalize=True, sort=False)
    
    # Contract success rates by category
    success_by_cat = bucket_data.groupby("contract_category").apply(
        lambda x: (x["rawscore"] > 0).mean()
    ).to_dict()
    
    contract_analysis.append({
        "player_type": "Human",
        "skill_bucket": bucket,
        "player_name": f"Human_{bucket}",
        "n_boards": len(bucket_data),
        "level_1_rate": level_dist.get(1.0, 0),
        "level_2_rate": level_dist.get(2.0, 0),
        "level_3_rate": level_dist.get(3.0, 0),
        "level_4_rate": level_dist.get(4.0, 0),
        "level_5_rate": level_dist.get(5.0, 0),
        "level_6_rate": level_dist.get(6.0, 0),
        "level_7_rate": level_dist.get(7.0, 0),
        "part_score_success": success_by_cat.get("Part-score", 0),
        "game_success": success_by_cat.get("Game", 0),
        "small_slam_success": success_by_cat.get("Small Slam", 0),
        "grand_slam_success": success_by_cat.get("Grand Slam", 0)
    })

# Analyze bots
for bot_name in bot_names:
    bot_data = all_rows[all_rows["player_name"] == bot_name]
    
    if len(bot_data) == 0:
        continue
    
    # Contract level distribution
    level_dist = bot_data["final_level"].value_counts(normalize=True, sort=False)
    
    # Contract success rates by category
    success_by_cat = bot_data.groupby("contract_category").apply(
        lambda x: (x["rawscore"] > 0).mean()
    ).to_dict()
    
    # Find bot's closest skill bucket
    bot_profile = bot_profiles_df[bot_profiles_df["bot_name"] == bot_name]
    closest_bucket = bot_profile["closest_skill_bucket"].iloc[0] if len(bot_profile) > 0 else "Unknown"
    
    contract_analysis.append({
        "player_type": "Bot",
        "skill_bucket": closest_bucket,
        "player_name": bot_name,
        "n_boards": len(bot_data),
        "level_1_rate": level_dist.get(1.0, 0),
        "level_2_rate": level_dist.get(2.0, 0),
        "level_3_rate": level_dist.get(3.0, 0),
        "level_4_rate": level_dist.get(4.0, 0),
        "level_5_rate": level_dist.get(5.0, 0),
        "level_6_rate": level_dist.get(6.0, 0),
        "level_7_rate": level_dist.get(7.0, 0),
        "part_score_success": success_by_cat.get("Part-score", 0),
        "game_success": success_by_cat.get("Game", 0),
        "small_slam_success": success_by_cat.get("Small Slam", 0),
        "grand_slam_success": success_by_cat.get("Grand Slam", 0)
    })

contract_level_analysis_df = pd.DataFrame(contract_analysis)
contract_level_analysis_df.to_parquet(os.path.join(OUT_DIR, "contract_level_distribution.parquet"), index=False)

print("Advanced analysis files created:")
print("- skill_bucket_profiles.parquet")
print("- bot_skill_profiles.parquet") 
print("- risk_analysis.parquet")
print("- contract_level_distribution.parquet")

print("Done. Outputs in:", OUT_DIR)
print("Data processing complete. Use create_advanced_visualizations.py to generate charts.")