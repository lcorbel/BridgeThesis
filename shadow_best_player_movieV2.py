#!/usr/bin/env python3
"""
Shadow best player movies V2: identical logic to shadow_best_player_movie.py but the
input comes directly from the humans/ and robot_replay/ folders (as described in
"data_readme (1).md").

- Humans input:
  - registrants.YYYYMMDD.tsv (team_id -> player)
  - detail.YYYYMMDD.tsv (per-deal rows with tourney_id, board_number, instance_id, movie, final_score)
  - summary.YYYYMMDD.tsv is not required (we compute player averages from detail), but if present it's ignored.

- Robot input:
  - robot_replay/louis.YYYYMMDD.<robot>[.<skill>] files containing RESULT/MOVIE pairs.

For each robot deal (tourney_id, board, instance), choose the best human player who played
that exact instance (highest global avg MP% from human detail across all days with at least
--min_instances rows). Then shadow the player's movie, step-by-step, asking the bot API to
predict the next action at each step, and record deviations.

Output:
- A semicolon-separated, fully-quoted CSV with the same columns as V1, saved by default to:
  results/shadowv2/shadow_best_player_movieV2.csv

Notes:
- Requires network access to the bot endpoints.
- Uses requests + xml parsing. Pandas is only used to read/upgrade existing CSV header for resume.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT.parent
HUMANS_DIR_DEFAULT = BASE_DIR / "humans"
ROBOTS_DIR_DEFAULT = BASE_DIR / "robot_replay"
OUTPUT_DIR_DEFAULT = BASE_DIR / "results" / "shadowv2"
OUTPUT_CSV_DEFAULT = OUTPUT_DIR_DEFAULT / "shadow_best_player_movieV2.csv"

# -------------------------- basic LIN parsing/building --------------------------

LIN_BID_RE = re.compile(r"mb\|([^|]+)\|", re.IGNORECASE)
LIN_PC_RE = re.compile(r"pc\|([shdcSHDC][akqjtAKQJT98765432])\|")
LIN_VULN_RE = re.compile(r"sv\|([^|])\|", re.IGNORECASE)
LIN_MD_DEALER_RE = re.compile(r"md\|([1-4])s", re.IGNORECASE)
LIN_MD_HANDS_RE = re.compile(r"md\|.([^|]+)\|", re.IGNORECASE)


def parse_lin_core(lin: str) -> Dict[str, object]:
    bids = [b.strip().upper().replace("!", "") for b in LIN_BID_RE.findall(lin)]
    # Normalize bids to history tokens: PASS->P, X/D->X, XX/R->XX
    bids_hist: List[str] = []
    for b in bids:
        u = b.upper()
        if u in ("P", "PASS"):
            bids_hist.append("P")
        elif u in ("X", "D"):
            bids_hist.append("X")
        elif u in ("XX", "R"):
            bids_hist.append("XX")
        else:
            bids_hist.append(u)

    cards = [c.upper() for c in LIN_PC_RE.findall(lin)]
    vuln = LIN_VULN_RE.findall(lin)
    vuln = (vuln[0].upper() if vuln else "-")
    dealer_num = LIN_MD_DEALER_RE.findall(lin)
    dealer = ("?SWNE"[int(dealer_num[0])] if dealer_num else "N")
    handsraw = LIN_MD_HANDS_RE.findall(lin)

    hands_bbo: List[str] = []
    if handsraw:
        raw = handsraw[0].strip().upper()
        parts = [p for p in raw.strip(',').split(',') if p != ""]
        if len(parts) != 4 or any(('S' not in p) for p in parts):
            if '.' in raw and ',' not in raw:
                dot_parts = [p for p in raw.split('.') if p]
                if len(dot_parts) >= 4:
                    accum: List[str] = []
                    current = ''
                    for seg in dot_parts:
                        if seg.startswith('S') and current:
                            accum.append(current)
                            current = seg
                        else:
                            current = (current + seg) if current else seg
                    if current:
                        accum.append(current)
                    if len(accum) == 4 and all(('S' in h and 'H' in h and 'D' in h and 'C' in h) for h in accum):
                        parts = accum
        # BBO LIN uses SWNE order after md|, convert to NESW expected by the rest of the code/API params
        if len(parts) == 4:
            parts_swne = parts
            parts = [parts_swne[2], parts_swne[3], parts_swne[0], parts_swne[1]]  # NESW
        hands_bbo = parts if len(parts) == 4 else []

    return {
        "hands_bbo": hands_bbo,  # list[str] NESW (converted from SWNE in md)
        "dealer": dealer,
        "vuln": vuln,
        "auction": bids_hist,
        "played_cards": cards,
    }


def bbo_to_dotted(hand_bbo: str) -> str:
    if not hand_bbo:
        return ""
    try:
        s_idx = hand_bbo.index("S"); h_idx = hand_bbo.index("H"); d_idx = hand_bbo.index("D"); c_idx = hand_bbo.index("C")
    except ValueError:
        return ""
    sp = hand_bbo[s_idx + 1 : h_idx]
    he = hand_bbo[h_idx + 1 : d_idx]
    di = hand_bbo[d_idx + 1 : c_idx]
    cl = hand_bbo[c_idx + 1 : ]
    return f"{sp}.{he}.{di}.{cl}"


# -------------------------- humans parsing --------------------------

def parse_date_from_filename(name: str) -> Optional[str]:
    m = re.search(r"(19|20)\d{6}", name)
    return m.group(0) if m else None


def read_registrants(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.is_file():
        return mapping
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header: Optional[List[str]] = None
        for row in reader:
            if not row:
                continue
            if header is None:
                header = row
                if "player" not in [c.lower() for c in header]:
                    header = ["tourney_id", "team_id", "player"]
                else:
                    continue
            try:
                if len(row) >= 3:
                    team_id = row[1].strip()
                    player = row[2].strip()
                    if team_id:
                        mapping[team_id] = player
            except Exception:
                continue
    return mapping


def stream_human_detail(path: Path, team_to_player: Dict[str, str]) -> Iterator[Dict[str, object]]:
    if not path.is_file():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header: Optional[List[str]] = None
        for row in reader:
            if not row:
                continue
            if header is None:
                header = row
                continue
            try:
                d = dict(zip(header, row)) if len(row) == len(header) else {}
                tourney_id = (d.get("tourney_id") or row[0]).strip()
                team_id = (d.get("team_id") or row[2]).strip()
                player = team_to_player.get(team_id, f"team_{team_id}" if team_id else "unknown")
                board_number = (d.get("board_number") or row[3]).strip()
                instance_id = (d.get("instance_id") or row[14] if len(row) > 14 else "").strip()
                result_str = (d.get("result") or row[7]).strip() if len(row) > 7 else ""
                rawscore = (d.get("rawscore") or row[8]).strip() if len(row) > 8 else ""
                final_score = (d.get("final_score") or row[9]).strip() if len(row) > 9 else ""
                movie = d.get("movie") or (row[10] if len(row) > 10 else "")
                yield {
                    "tourney_id": tourney_id,
                    "team_id": team_id,
                    "player": player,
                    "board_number": board_number,
                    "instance_id": instance_id,
                    "result_str": result_str,
                    "rawscore": _safe_float(rawscore),
                    "final_score": _safe_float(final_score),
                    "movie": movie,
                }
            except Exception:
                continue


def build_human_indexes(humans_dir: Path) -> Tuple[Dict[Tuple[str, str, str], List[Tuple[str, str]]], Dict[str, Tuple[int, float]]]:
    """Return:
    - instance_map: (tourney_id, board_number, instance_id) -> list of (player_label, player_movie)
    - player_stats: player_label -> (count_rows, avg_final_score) from detail
    """
    # date -> team mapping
    registrants_by_date: Dict[str, Dict[str, str]] = {}
    for name in os.listdir(humans_dir):
        if name.startswith("registrants.") and name.endswith(".tsv"):
            date = parse_date_from_filename(name)
            if not date:
                continue
            registrants_by_date[date] = read_registrants(humans_dir / name)

    instance_map: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
    # player aggregation from detail final_score
    player_sum: Dict[str, float] = {}
    player_cnt: Dict[str, int] = {}

    for name in sorted(os.listdir(humans_dir)):
        if not (name.startswith("detail.") and name.endswith(".tsv")):
            continue
        date = parse_date_from_filename(name) or ""
        team_to_player = registrants_by_date.get(date, {})
        path = humans_dir / name
        for rec in stream_human_detail(path, team_to_player):
            tourney_id = str(rec.get("tourney_id") or "").strip()
            board_number = str(rec.get("board_number") or "").strip()
            instance_id = str(rec.get("instance_id") or "").strip()
            player = str(rec.get("player") or "unknown").strip()
            movie = str(rec.get("movie") or "").strip()
            if not (tourney_id and board_number and instance_id and movie):
                continue
            key = (tourney_id, board_number, instance_id)
            instance_map.setdefault(key, []).append((player, movie))
            fs = rec.get("final_score")
            if isinstance(fs, (int, float)):
                player_sum[player] = player_sum.get(player, 0.0) + float(fs)
                player_cnt[player] = player_cnt.get(player, 0) + 1

    player_stats: Dict[str, Tuple[int, float]] = {}
    for p, cnt in player_cnt.items():
        avg = player_sum.get(p, 0.0) / cnt if cnt > 0 else 0.0
        player_stats[p] = (cnt, avg)
    return instance_map, player_stats


# -------------------------- robot replay parsing --------------------------
RESULT_TITLE_RE = re.compile(r"title=(.*?)\s+board=", re.IGNORECASE)


def parse_result_line(line: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    s = line.strip()
    if not s.startswith("RESULT"):
        return out
    rest = s[len("RESULT"):].strip()
    title = None
    m = RESULT_TITLE_RE.search(rest)
    if m:
        title = m.group(1).strip()
        rest = rest[: m.start()] + rest[m.end() :]
    out["title"] = title or ""
    mp = None
    m2 = re.search(r"mp%=([^\s]+)", rest)
    if m2:
        try:
            mp = float(m2.group(1))
        except Exception:
            mp = None
        rest = rest.replace(m2.group(0), "")
    out["mp_pct"] = mp
    keys = ["bot_id", "tourney", "board", "instance", "result", "rawscore"]
    for k in keys:
        m3 = re.search(rf"\b{k}=([^\s]+)", rest)
        if m3:
            val = m3.group(1)
            if k in ("board", "instance"):
                try:
                    out[k] = int(val)
                except Exception:
                    out[k] = None
            elif k in ("rawscore",):
                out[k] = _safe_float(val)
            else:
                out[k] = val
    return out


def parse_movie_line(line: str) -> str:
    s = line.strip()
    if not s.startswith("MOVIE"):
        return ""
    rest = s[len("MOVIE"):].strip()
    return rest


def stream_robot_replay(path: Path, date: str, robot: str, skill: Optional[str]) -> Iterator[Dict[str, object]]:
    if not path.is_file():
        return
    pending_result: Optional[Dict[str, object]] = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if not raw.strip():
                continue
            if raw.startswith("RESULT"):
                pending_result = parse_result_line(raw)
                continue
            if raw.startswith("MOVIE"):
                movie = parse_movie_line(raw)
                if pending_result is None:
                    continue
                rec: Dict[str, object] = {
                    "date": date,
                    "robot": robot,
                    "skill": skill or "",
                    "movie": movie,
                }
                rec.update(pending_result)
                yield rec
                pending_result = None


def iter_robot_records(robots_dir: Path) -> Iterator[Dict[str, object]]:
    for name in sorted(os.listdir(robots_dir)):
        if not name.startswith("louis."):
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        date = parts[1]
        robot = parts[2]
        skill = parts[3] if len(parts) >= 4 else ""
        path = robots_dir / name
        bot_label = None
        if robot == "argine":
            bot_label = "argine"
        elif robot == "gib":
            if skill in ("basic", "advanced"):
                bot_label = f"gib_{skill}"
            else:
                continue
        elif robot == "ben":
            if skill:
                bot_label = f"ben_{skill}"
            else:
                continue
        else:
            continue
        for rec in stream_robot_replay(path, date, robot, skill):
            tourney_id = str(rec.get("tourney") or "").strip()
            board = rec.get("board")
            instance = rec.get("instance")
            movie = str(rec.get("movie") or "").strip()
            if not (tourney_id and isinstance(board, int) and isinstance(instance, int) and movie):
                continue
            yield {
                "tourney_id": tourney_id,
                "board_number": str(board),
                "instance_id": str(instance),
                "bot": bot_label,
                "bot_movie": movie,
                "vuln": None,  # determined from player movie later
            }


# -------------------------- contract parsing (for completeness) --------------------------
CONTRACT_RE = re.compile(r"^([1-7])([CDHSN])([NESW])([=]|[+\-]\d+)$", re.IGNORECASE)


def parse_contract(result_str: str) -> Dict[str, object]:
    out = {
        "contract_level": None,
        "contract_denom": "",
        "declarer": "",
        "tricks_delta": None,
        "tricks_made": None,
        "made": None,
    }
    if not result_str:
        return out
    rs = result_str.strip().upper()
    m = CONTRACT_RE.match(rs)
    if not m:
        return out
    level = int(m.group(1))
    denom = m.group(2)
    decl = m.group(3)
    delta_s = m.group(4)
    delta = 0 if delta_s == "=" else int(delta_s)
    base = 6 + level
    tricks = base + delta
    out.update({
        "contract_level": level,
        "contract_denom": denom,
        "declarer": decl,
        "tricks_delta": delta,
        "tricks_made": tricks,
        "made": int(delta >= 0),
    })
    return out


# -------------------------- POV and turn computation --------------------------
SEATS = ('N','E','S','W')
IDX = {s:i for i,s in enumerate(SEATS)}
RANKS = {c: i for i, c in enumerate(list('23456789TJQKA'), start=2)}


def _seat_to_act_bidding(dealer: str, bids_count: int) -> str:
    return SEATS[(IDX[dealer] + bids_count) % 4]


def _is_contract_bid(tok: str) -> bool:
    t = tok.upper()
    return len(t) >= 2 and t[0] in "1234567" and t[1] in "CDHSN"


def _dealer_index(seat: str) -> int:
    try:
        return SEATS.index(seat.upper())
    except Exception:
        return 0


def _side_of_idx(idx: int) -> str:
    return "NS" if idx % 2 == 0 else "EW"


def _compute_declarer(dealer: str, bids: List[str]) -> str:
    d_idx = _dealer_index(dealer)
    final_idx: Optional[int] = None
    final_strain: Optional[str] = None
    for i, tok in enumerate(bids):
        if _is_contract_bid(tok):
            final_idx = i
            final_strain = tok[1].upper()
    if final_idx is None or not final_strain:
        return dealer.upper()
    final_seat_idx = (d_idx + final_idx) % 4
    final_side_parity = final_seat_idx % 2
    for i, tok in enumerate(bids):
        if _is_contract_bid(tok) and tok[1].upper() == final_strain:
            seat_idx = (d_idx + i) % 4
            if seat_idx % 2 == final_side_parity:
                return SEATS[seat_idx]
    return SEATS[final_seat_idx]


def _auction_complete(bids: List[str]) -> bool:
    if not bids:
        return False
    i = len(bids) - 1
    trailing_passes = 0
    while i >= 0 and bids[i] == 'P':
        trailing_passes += 1
        i -= 1
    if trailing_passes < 3:
        return False
    return any(_is_contract_bid(b) for b in bids[:i+1])


def _final_contract_strain(bids: List[str]) -> Optional[str]:
    for tok in reversed(bids):
        if _is_contract_bid(tok):
            return tok[1].upper()
    return None


def _left_of(seat: str) -> str:
    idx = _dealer_index(seat)
    return SEATS[(idx + 1) % 4]


def _partner_of(seat: str) -> str:
    idx = _dealer_index(seat)
    return SEATS[(idx + 2) % 4]


def _hand_has_card(hand_bbo: str, card: str) -> bool:
    if not hand_bbo or len(card) < 2:
        return False
    suit = card[0].upper()
    rank = card[1].upper()
    try:
        s_idx = hand_bbo.index('S'); h_idx = hand_bbo.index('H'); d_idx = hand_bbo.index('D'); c_idx = hand_bbo.index('C')
    except ValueError:
        return False
    if suit == 'S':
        seg = hand_bbo[s_idx+1:h_idx]
    elif suit == 'H':
        seg = hand_bbo[h_idx+1:d_idx]
    elif suit == 'D':
        seg = hand_bbo[d_idx+1:c_idx]
    else:
        seg = hand_bbo[c_idx+1:]
    return rank in seg


def _seat_of_card(hands_bbo: List[str], card: str) -> Optional[str]:
    if len(hands_bbo) != 4:
        return None
    for seat, hand in zip(SEATS, hands_bbo):
        if _hand_has_card(hand, card):
            return seat
    return None


def _trick_winner(trick: List[str], leader: str, trump: Optional[str]) -> str:
    led_suit = trick[0][0].upper()
    leader_idx = _dealer_index(leader)
    candidates: List[Tuple[int, str]] = []
    if trump and trump != 'N' and any(c[0].upper() == trump for c in trick):
        for i, c in enumerate(trick):
            if c[0].upper() == trump:
                candidates.append((i, c))
    else:
        for i, c in enumerate(trick):
            if c[0].upper() == led_suit:
                candidates.append((i, c))
    win_rel, win_card = max(candidates, key=lambda x: RANKS.get(x[1][1].upper(), 0))
    return SEATS[(leader_idx + win_rel) % 4]


def _is_auction_token(tok: str) -> bool:
    t = tok.upper()
    return _is_contract_bid(t) or t in ("P", "X", "XX")


def _compute_seat_to_act(dealer: str, hands_bbo: List[str], history_tokens: List[str]) -> str:
    bids = [t for t in history_tokens if _is_auction_token(t)]
    plays = [t for t in history_tokens if not _is_auction_token(t)]

    d_idx = _dealer_index(dealer)

    if not _auction_complete(bids):
        seat_idx = (d_idx + len(bids)) % 4
        return SEATS[seat_idx]

    if plays:
        leader = _seat_of_card(hands_bbo, plays[0])
    else:
        leader = None
    if not leader:
        declarer = _compute_declarer(dealer, bids)
        leader = _left_of(declarer)

    if not plays:
        return leader

    trump = _final_contract_strain(bids)

    completed = len(plays) // 4
    current_leader = leader
    for t in range(completed):
        trick = plays[t*4:(t+1)*4]
        winner = _trick_winner(trick, current_leader, trump)
        current_leader = winner

    offset = len(plays) % 4
    return SEATS[(_dealer_index(current_leader) + offset) % 4]


# -------------------------- bot endpoints --------------------------
class BotEndpoint:
    def __init__(self, base_url: str, style: Optional[str] = None, extra_params: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.style = style
        self.extra_params = extra_params or {}
    def params(self) -> Dict[str, str]:
        p = dict(self.extra_params)
        if self.style:
            p["botstyle"] = self.style
            # Argine uses 'advanced' but also expects some defaults
        return p


def endpoint_for_bot(bot_label: str) -> Optional[BotEndpoint]:
    if bot_label.startswith("ben_"):
        style = bot_label.split("_", 1)[1]
        return BotEndpoint("https://ben.dev.cl.bridgebase.com/u_bm/robot.php", style, {"bm": "y"})
    if bot_label == "gib_basic":
        return BotEndpoint("https://gibrest.bridgebase.com/u_bm/robot.php", "basic")
    if bot_label == "gib_advanced":
        return BotEndpoint("https://gibrest.bridgebase.com/u_bm/robot.php", "advanced")
    if bot_label == "argine":
        return BotEndpoint(
            "https://gibrest.bridgebase.com/argine/robot.php",
            "advanced",
            {"nsConv": "2/1", "ewConv": "2/1", "ac": "y"},
        )
    return None


def _bot_family_from_endpoint(endpoint: BotEndpoint) -> str:
    url = (endpoint.base_url or "").lower()
    if "/argine/" in url:
        return "argine"
    if "ben.dev.cl.bridgebase.com" in url:
        return "ben"
    if "/u_bm/" in url and "gibrest.bridgebase.com" in url:
        return "gib"
    return "other"


# --- Vulnerability helpers ---
_DEF_V_SCHEDULE = [0,1,2,3, 1,2,3,0, 2,3,0,1, 3,0,1,2]


def _extract_sv_code_from_movie(movie: Optional[str]) -> Optional[int]:
    if not isinstance(movie, str) or not movie:
        return None
    m = re.search(r"sv\|([^|])\|", movie, flags=re.IGNORECASE)
    if not m:
        return None
    v = m.group(1).upper()
    if v in ("0","1","2","3"):
        return int(v)
    if v in ("-","O"):
        return 0
    if v == "N":
        return 1
    if v == "E":
        return 2
    if v == "B":
        return 3
    return None


def _vuln_code_from_letter_or_dash(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    vu = str(v).strip().upper()
    if vu in ("0","1","2","3"):
        return int(vu)
    if vu in ("-","O"):
        return 0
    if vu == "N":
        return 1
    if vu == "E":
        return 2
    if vu == "B":
        return 3
    return None


def _vuln_code_from_board(board_num: object) -> int:
    try:
        b = int(board_num)
        if b <= 0:
            return 0
        return _DEF_V_SCHEDULE[(b - 1) % 16]
    except Exception:
        return 0

_DEF_V_MAP_ARGINE = {0: '-', 1: 'N', 2: 'E', 3: 'B'}
_DEF_V_MAP_GIB_BEN = {0: '-', 1: 'N', 2: 'E', 3: 'B'}


def _shadow_endpoint_with_ben_v(endpoint: BotEndpoint, vuln: str, bot_label: str,
                                 player_movie: Optional[str] = None,
                                 board_number: Optional[object] = None) -> BotEndpoint:
    sv_code = _extract_sv_code_from_movie(player_movie)
    if sv_code is None:
        sv_code = _vuln_code_from_letter_or_dash(vuln)
    if sv_code is None:
        sv_code = _vuln_code_from_board(board_number)

    if bot_label == 'argine':
        v_param = _DEF_V_MAP_ARGINE[sv_code]
        extra = dict(endpoint.extra_params)
        extra.setdefault('ac', 'y')
        extra.setdefault('nsConv', '2/1')
        extra.setdefault('ewConv', '2/1')
        extra['v'] = v_param
        return BotEndpoint(endpoint.base_url, endpoint.style, extra)

    if bot_label.startswith('ben_'):
        v_param = _DEF_V_MAP_GIB_BEN[sv_code]
        extra = dict(endpoint.extra_params)
        extra['v'] = v_param
        return BotEndpoint(endpoint.base_url, endpoint.style, extra)

    if bot_label.startswith('gib_'):
        v_param = _DEF_V_MAP_GIB_BEN[sv_code]
        extra = dict(endpoint.extra_params)
        extra['v'] = v_param
        return BotEndpoint(endpoint.base_url, endpoint.style, extra)

    return endpoint


# -------------------------- API call --------------------------

def _map_history_tokens_for_api(_family: str, history_tokens: List[str]) -> List[str]:
    mapped: List[str] = []
    for t in history_tokens:
        tu = t.upper()
        if tu == 'D':
            mapped.append('X')
        elif tu == 'R':
            mapped.append('XX')
        else:
            mapped.append(tu)
    return mapped


def call_bot_next(endpoint: BotEndpoint, dealer: str, vuln: str, hands_bbo: List[str], history_tokens: List[str], timeout: float = 10.0) -> Tuple[str, Optional[str]]:
    pov_final = _compute_seat_to_act(dealer, hands_bbo, history_tokens)
    bids_only = [t for t in history_tokens if _is_auction_token(t)]
    plays_only = [t for t in history_tokens if not _is_auction_token(t)]
    family = _bot_family_from_endpoint(endpoint)
    if _auction_complete(bids_only):
        declarer = _compute_declarer(dealer, bids_only)
        if declarer and family in ("gib") and pov_final == _partner_of(declarer):
            pov_final = declarer

    hands_dotted = [bbo_to_dotted(h).upper() for h in hands_bbo]
    if len(hands_dotted) != 4 or any(not x for x in hands_dotted):
        raise ValueError(f"Invalid hands for API call: {hands_bbo}")
    n, e, s, w = hands_dotted[0], hands_dotted[1], hands_dotted[2], hands_dotted[3]

    h_tokens_api = _map_history_tokens_for_api(family, history_tokens)
    h_str = "-".join(h_tokens_api)

    params = {
        "pov": pov_final,
        "d": (dealer or "N").upper(),
        "n": n,
        "e": e,
        "s": s,
        "w": w,
        "h": h_str,
        "sc": "MP",
    }
    params.update(endpoint.params())

    prepared = requests.Request("GET", endpoint.base_url, params=params).prepare()

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            with requests.Session() as s:
                resp = s.send(prepared, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            root = ET.fromstring(text)
            break
        except requests.HTTPError as he:
            status = he.response.status_code if he.response is not None else 0
            if status >= 500 and attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            url = prepared.url or endpoint.base_url
            body = he.response.text if he.response is not None else ""
            snippet = body[:300].replace("\n", " ") if isinstance(body, str) else ""
            raise RuntimeError(f"Bot API HTTP {status}: {url} :: {snippet}") from he
        except ET.ParseError as pe:
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            url = prepared.url or endpoint.base_url
            raise RuntimeError(f"Bot API invalid XML: {url} :: {type(pe).__name__}: {pe}") from pe
        except Exception as e:
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            url = prepared.url or endpoint.base_url
            raise RuntimeError(f"Bot API call failed: {url} :: {type(e).__name__}: {e}") from e

    r = root.find(".//r")
    if r is None:
        err_attr = root.attrib.get("error") if hasattr(root, 'attrib') else None
        rc_attr = root.attrib.get("rc") if hasattr(root, 'attrib') else None
        err = root.findtext('.//error') or root.findtext('.//msg') or root.findtext('.//message') or err_attr
        url = prepared.url or endpoint.base_url
        if err or rc_attr:
            details = []
            if err:
                details.append(str(err))
            if rc_attr:
                details.append(f"rc={rc_attr}")
            raise RuntimeError(f"Bot API error: {url} :: {'; '.join(details)}")
        snippet = text[:300].replace("\n", " ") if isinstance(text, str) else ""
        raise RuntimeError(f"Bot API missing <r> element: {url} :: body: {snippet}")
    rtype = r.get("type")
    if rtype == "bid":
        raw_bid = (r.get("bid") or r.get("call") or r.get("b") or "").upper()
        if raw_bid in ("P", "PASS"):
            tok = "P"
        elif raw_bid in ("X", "D"):
            tok = "X"
        elif raw_bid in ("XX", "R"):
            tok = "XX"
        else:
            tok = raw_bid
        return ("BID", tok)
    if rtype == "play":
        card = (r.get("card") or r.get("pc") or "").upper()
        return ("PLAY", card)
    if rtype == "result" or rtype == "claim":
        return ("RESULT", None)
    bid_attr = (r.get("bid") or r.get("call") or r.get("b"))
    card_attr = (r.get("card") or r.get("pc"))
    if bid_attr:
        bid = bid_attr.upper()
        if bid == "P" or bid == "PASS":
            tok = "P"
        elif bid in ("X", "D"):
            tok = "X"
        elif bid in ("XX", "R"):
            tok = "XX"
        else:
            tok = bid
        return ("BID", tok)
    if card_attr:
        return ("PLAY", card_attr.upper())
    url = prepared.url or endpoint.base_url
    raise RuntimeError(f"Bot API unknown response: {url}")


# -------------------------- output helpers (semicolon CSV + resume) --------------------------
DEV_COL = "nb of steps deviating between bot prediction an player movie"
DEV_COL_BIDDING = "Nb of deviation in bidding phase"
DEV_COL_PLAY_EXCL_OPENING = "nb of deviation in card play phase (exclude oopening lead)"
DEV_COL_OPENING = "Nb of deviation in opening lead"
BASE_COLS = [
    DEV_COL,
    DEV_COL_BIDDING,
    DEV_COL_PLAY_EXCL_OPENING,
    DEV_COL_OPENING,
    "tourney_id",
    "board_number",
    "instance_id",
    "bot",
    "player_label",
    "bot_movie",
    "player_movie",
]


def build_output_columns(max_steps: int) -> List[str]:
    cols = list(BASE_COLS)
    for s in range(2, max_steps + 2):
        cols.append(f"bot prediction step {s}")
        cols.append(f"actual player movie step {s}")
    return cols


def _q_csv(s: str) -> str:
    s2 = s.replace(",", ".")
    return '"' + s2.replace('"', '""') + '"'


def open_output_for_stream(path: Path, header_cols: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() and path.stat().st_size > 0 else "w"
    f = path.open(mode, encoding="utf-8", newline="")
    if mode == "w":
        f.write(";".join(_q_csv(c) for c in header_cols) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return f


def write_row_stream(f, header_cols: List[str], row: Dict[str, object]) -> None:
    vals: List[str] = []
    for k in header_cols:
        v = row.get(k, "")
        vals.append(_q_csv("" if v is None else str(v)))
    f.write(";".join(vals) + "\n")
    f.flush()
    try:
        os.fsync(f.fileno())
    except Exception:
        pass


def read_existing_output_header(path: Path) -> Optional[List[str]]:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        hdr_df = pd.read_csv(path, sep=';', engine='python', quotechar='"', nrows=0)
        return list(hdr_df.columns)
    except Exception:
        return None


def ensure_output_has_columns(path: Path, required_cols: List[str]) -> Optional[List[str]]:
    try:
        cols = read_existing_output_header(path)
        if not cols:
            return cols
        missing = [c for c in required_cols if c not in cols]
        if not missing:
            return cols
        with path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            return cols
        header_line = lines[0].rstrip('\n')
        header_line = header_line + ''.join(f';"{m}"' for m in missing) + "\n"
        new_lines = [header_line]
        for ln in lines[1:]:
            ln_stripped = ln.rstrip('\n')
            if ln_stripped == '':
                new_lines.append(ln)
            else:
                new_lines.append(ln_stripped + ';' + ';'.join(['""'] * len(missing)) + '\n')
        with path.open('w', encoding='utf-8', newline='') as f:
            f.writelines(new_lines)
        return cols + missing
    except Exception:
        return read_existing_output_header(path)


def get_last_processed_keys(path: Path) -> Optional[Tuple[str, str, str, str]]:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        df_out = pd.read_csv(path, sep=';', engine='python', quotechar='"')
        if df_out.empty:
            return None
        last = df_out.tail(1).iloc[0]
        t = str(last.get('tourney_id')) if 'tourney_id' in df_out.columns else None
        b = str(last.get('board_number')) if 'board_number' in df_out.columns else None
        i = str(last.get('instance_id')) if 'instance_id' in df_out.columns else None
        bot = str(last.get('bot')) if 'bot' in df_out.columns else None
        def norm(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            sx = str(x)
            return None if sx.lower() == 'nan' else sx
        return (norm(t) or '', norm(b) or '', norm(i) or '', norm(bot) or '')
    except Exception:
        return None


# -------------------------- misc utils --------------------------

def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        xs = str(x).strip()
        if xs == '' or xs.upper() in ('NULL','NA','N/A','NONE'):
            return None
        return float(xs)
    except Exception:
        return None


# -------------------------- alignment helpers --------------------------

def _maybe_infer_dealer_from_openinglead(hands_bbo: List[str], dealer: str, bids: List[str], plays: List[str], debug: bool = False) -> str:
    if not plays:
        return dealer
    leader = _seat_of_card(hands_bbo, plays[0])
    if not leader:
        return dealer
    matches: List[str] = []
    for cand in SEATS:
        dec = _compute_declarer(cand, bids)
        if _left_of(dec) == leader:
            matches.append(cand)
    if len(matches) == 1 and matches[0] != dealer:
        if debug:
            try:
                print(f"DEBUG dealer adjusted {dealer} -> {matches[0]} based on opening lead {plays[0]} by {leader} and declarer {_compute_declarer(matches[0], bids)}")
            except Exception:
                pass
        return matches[0]
    return dealer


def _ensure_hands_alignment(hands_bbo: List[str], dealer: str, bids: List[str], plays: List[str], debug: bool = False) -> List[str]:
    if not hands_bbo or len(hands_bbo) != 4 or not plays:
        return hands_bbo
    expected_leader = _left_of(_compute_declarer(dealer, bids))
    leader_cur = _seat_of_card(hands_bbo, plays[0])
    if leader_cur == expected_leader:
        return hands_bbo
    alt = [hands_bbo[2], hands_bbo[3], hands_bbo[0], hands_bbo[1]]
    leader_alt = _seat_of_card(alt, plays[0])
    if leader_alt == expected_leader:
        if debug:
            try:
                print(f"DEBUG hands alignment adjusted to match opening leader {plays[0]}: {leader_cur} -> {leader_alt}")
            except Exception:
                pass
        return alt
    return hands_bbo


# -------------------------- selection helpers --------------------------

def select_best_player_for_instance(candidates: List[Tuple[str, str]], player_stats: Dict[str, Tuple[int, float]], min_instances: int, force_player_label: str = "") -> Optional[Tuple[str, str]]:
    if not candidates:
        return None
    if force_player_label:
        for pl, mv in candidates:
            if pl == force_player_label:
                return (mv, pl)
        return None
    elig: List[Tuple[str, float, str]] = []
    for pl, mv in candidates:
        cnt_avg = player_stats.get(pl)
        if not cnt_avg:
            continue
        cnt, avg = cnt_avg
        if cnt < min_instances:
            continue
        elig.append((pl, avg, mv))
    if not elig:
        return None
    pl, _avg, mv = max(elig, key=lambda t: t[1])
    return (mv, pl)


# -------------------------- main --------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Shadow best player movie V2 (humans + robot_replay inputs)")
    ap.add_argument("--humans_dir", default=str(HUMANS_DIR_DEFAULT), help="Path to humans directory")
    ap.add_argument("--robots_dir", default=str(ROBOTS_DIR_DEFAULT), help="Path to robot_replay directory")
    ap.add_argument("--output", default=str(OUTPUT_CSV_DEFAULT), help="Output CSV path (semicolon-separated)")
    ap.add_argument("--min_instances", type=int, default=5, help="Minimum human detail rows required for a player to be eligible")
    ap.add_argument("--limit_rows", type=int, default=0, help="Optional limit of output rows to produce (0 = all)")
    ap.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout per bot call in seconds")
    ap.add_argument("--debug", choices=["Y", "N"], default="N", help="If Y, print parsed tokens and details")
    ap.add_argument("--force_player_label", default="", help="If set, require this player on each instance (must have played that instance)")
    ap.add_argument("--max_steps", type=int, default=200, help="Max step label to include in the output header")
    args = ap.parse_args()

    humans_dir = Path(args.humans_dir)
    robots_dir = Path(args.robots_dir)
    out_path = Path(args.output)

    print("[INFO] Indexing humans ...", file=sys.stderr)
    instance_map, player_stats = build_human_indexes(humans_dir)

    desired_header_cols = build_output_columns(args.max_steps)
    required_cols = list(BASE_COLS)
    existing_header_cols = ensure_output_has_columns(out_path, required_cols)
    if existing_header_cols is None:
        existing_header_cols = read_existing_output_header(out_path)
    header_cols = existing_header_cols if existing_header_cols else desired_header_cols
    out_f = open_output_for_stream(out_path, header_cols)

    last_keys = get_last_processed_keys(out_path)
    resume_after = None
    if last_keys:
        resume_after = tuple(last_keys)  # (tourney, board, instance, bot)

    processed = 0
    print("[INFO] Processing robot replays ...", file=sys.stderr)
    for rec in iter_robot_records(robots_dir):
        t = rec["tourney_id"]
        b = rec["board_number"]
        i = rec["instance_id"]
        bot = rec["bot"]
        bot_movie = rec["bot_movie"]
        # Resume logic: skip until after last processed keys
        if resume_after:
            if (t, b, i, bot) == resume_after:
                resume_after = None
            else:
                continue
            # After matching the last row, continue to next record (do not reprocess matched one)
            continue

        candidates = instance_map.get((t, b, i), [])
        sel = select_best_player_for_instance(candidates, player_stats, args.min_instances, args.force_player_label)
        if not sel:
            continue
        player_movie, player_label = sel
        core = parse_lin_core(player_movie)
        hands_bbo = core["hands_bbo"]  # type: ignore
        dealer = core["dealer"]  # type: ignore
        vuln = core["vuln"]  # type: ignore
        bids = core["auction"]  # type: ignore
        cards = core["played_cards"]  # type: ignore
        hands_bbo = _ensure_hands_alignment(hands_bbo, dealer, bids, cards, debug=(args.debug == "Y"))
        dealer_orig = dealer
        dealer_inferred = _maybe_infer_dealer_from_openinglead(hands_bbo, dealer_orig, bids, cards, debug=(args.debug == "Y"))
        tokens: List[str] = list(bids) + list(cards)
        if len(tokens) < 2:
            continue

        endpoint = endpoint_for_bot(bot)
        if endpoint is None:
            continue
        endpoint = _shadow_endpoint_with_ben_v(endpoint, vuln, bot, player_movie=player_movie, board_number=b)

        out_row: Dict[str, object] = {
            "tourney_id": t,
            "board_number": b,
            "instance_id": i,
            "bot": bot,
            "player_label": player_label,
            "bot_movie": bot_movie,
            "player_movie": player_movie,
        }

        deviations = 0
        dev_bidding = 0
        dev_opening = 0
        dev_play_excl_open = 0

        for step in range(1, len(tokens)):
            hist = tokens[:step]
            hist_bids = [tkn for tkn in hist if _is_auction_token(tkn)]
            hist_plays = [tkn for tkn in hist if not _is_auction_token(tkn)]
            use_dealer = dealer_orig if len(hist_plays) == 0 else dealer_inferred
            try:
                kind, tok = call_bot_next(endpoint, use_dealer, vuln, hands_bbo, hist, timeout=args.timeout)
            except Exception as e:
                # best-effort error reporting
                try:
                    hands_dotted = [bbo_to_dotted(h).upper() for h in hands_bbo]
                    n_hand, e_hand, s_hand, w_hand = hands_dotted
                    fam = _bot_family_from_endpoint(endpoint)
                    h_str = "-".join(_map_history_tokens_for_api(fam, hist))
                    pov_final = _compute_seat_to_act(use_dealer, hands_bbo, hist)
                    bids_only = [tkn for tkn in hist if _is_auction_token(tkn)]
                    if _auction_complete(bids_only):
                        declarer = _compute_declarer(use_dealer, bids_only)
                        fam = _bot_family_from_endpoint(endpoint)
                        if declarer and fam in ("gib") and pov_final == _partner_of(declarer):
                            pov_final = declarer
                    params = {
                        "pov": pov_final,
                        "d": (use_dealer or "N").upper(),
                        "n": n_hand,
                        "e": e_hand,
                        "s": s_hand,
                        "w": w_hand,
                        "h": h_str,
                        "sc": "MP",
                    }
                    params.update(endpoint.params())
                    prepared = requests.Request("GET", endpoint.base_url, params=params).prepare()
                    url = prepared.url
                except Exception:
                    url = f"{endpoint.base_url}?<failed-to-reconstruct-query>"
                print(f"ERROR while calling bot endpoint for {bot}. Query: {url}", file=sys.stderr)
                print(f"{type(e).__name__}: {e}", file=sys.stderr)
                if args.debug == "Y":
                    try:
                        print(f"DEBUG failing hist={hist}")
                    except Exception:
                        pass
                pred = ""
                actual = tokens[step]
            else:
                pred = "RESULT" if (kind == "RESULT" and tok is None) else (tok or "")
                actual = tokens[step]

            if pred != actual:
                deviations += 1
                if not _auction_complete(hist_bids):
                    dev_bidding += 1
                elif len(hist_plays) == 0:
                    dev_opening += 1
                else:
                    dev_play_excl_open += 1

            out_row[f"bot prediction step {step+1}"] = pred
            out_row[f"actual player movie step {step+1}"] = actual

        out_row[DEV_COL] = deviations
        out_row[DEV_COL_BIDDING] = dev_bidding
        out_row[DEV_COL_OPENING] = dev_opening
        out_row[DEV_COL_PLAY_EXCL_OPENING] = dev_play_excl_open

        write_row_stream(out_f, header_cols, out_row)
        processed += 1
        if args.limit_rows and processed >= args.limit_rows:
            break

    try:
        out_f.close()
    except Exception:
        pass
    print(f"Wrote (streaming) to {out_path}")


if __name__ == "__main__":
    main()
