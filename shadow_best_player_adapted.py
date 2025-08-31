#!/usr/bin/env python3
"""
Shadow best player movies - Dataset Analysis Adapted Version

Adapted from shadow_best_player_movieV2.py to work with the dataset_analysis merged files:
- detail_20250701_20250710_merged.tsv (human data)
- registrants_20250701_20250710_merged.tsv (team to player mapping)
- louis_20250701_20250710_*_merged (bot files)

For each bot deal (tourney_id, board, instance), finds the best human player who played
that exact instance (highest MP% from human detail) and shadows their movie step-by-step,
asking the bot API to predict the next action and recording deviations.

Output: shadow_analysis.parquet with debugging info for each tournament/board/instance
"""

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# Bot endpoint configurations
class BotEndpoint:
    def __init__(self, base_url: str, style: Optional[str] = None, extra_params: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.style = style
        self.extra_params = extra_params or {}
    
    def params(self) -> Dict[str, str]:
        p = dict(self.extra_params)
        if self.style:
            p["botstyle"] = self.style
        return p

def endpoint_for_bot(bot_name: str) -> Optional[BotEndpoint]:
    """Map bot names from our dataset to API endpoints."""
    if "ben_advanced" in bot_name:
        return BotEndpoint("https://ben.dev.cl.bridgebase.com/u_bm/robot.php", "advanced", {"bm": "y"})
    elif "ben_intermediate" in bot_name:
        return BotEndpoint("https://ben.dev.cl.bridgebase.com/u_bm/robot.php", "intermediate", {"bm": "y"})
    elif "ben_beginner" in bot_name:
        return BotEndpoint("https://ben.dev.cl.bridgebase.com/u_bm/robot.php", "beginner", {"bm": "y"})
    elif "ben_novice" in bot_name:
        return BotEndpoint("https://ben.dev.cl.bridgebase.com/u_bm/robot.php", "novice", {"bm": "y"})
    elif "gib_basic" in bot_name:
        return BotEndpoint("https://gibrest.bridgebase.com/u_bm/robot.php", "basic")
    elif "gib_advanced" in bot_name:
        return BotEndpoint("https://gibrest.bridgebase.com/u_bm/robot.php", "advanced")
    elif "argine" in bot_name:
        return BotEndpoint(
            "https://gibrest.bridgebase.com/argine/robot.php",
            "advanced",
            {"nsConv": "2/1", "ewConv": "2/1", "ac": "y"},
        )
    return None

# Bridge constants and utilities
SEATS = ["N", "E", "S", "W"]
SUITS = ["C", "D", "H", "S"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
IDX = {s: i for i, s in enumerate(SEATS)}
RANK_VALUES = {c: i for i, c in enumerate(RANKS, start=2)}

# LIN parsing regex patterns
LIN_BID_RE = re.compile(r"mb\|([^|]+)\|", re.IGNORECASE)
LIN_PC_RE = re.compile(r"pc\|([shdcSHDC][akqjtAKQJT98765432])\|")
LIN_VULN_RE = re.compile(r"sv\|([^|])\|", re.IGNORECASE)
LIN_MD_DEALER_RE = re.compile(r"md\|([1-4])s", re.IGNORECASE)
LIN_MD_HANDS_RE = re.compile(r"md\|.([^|]+)\|", re.IGNORECASE)

def parse_lin_tokens(movie: str) -> List[str]:
    """Parse LIN movie into tokens."""
    if not movie or not isinstance(movie, str):
        return []
    
    tokens = []
    parts = movie.split("|")
    for i in range(0, len(parts)-1, 2):
        if i+1 < len(parts):
            tag = parts[i].strip()
            value = parts[i+1].strip()
            if tag == "mb":  # bidding
                tokens.append(value)
            elif tag == "pc":  # play card
                tokens.append(value)
    return tokens

def parse_lin_core(lin: str) -> Dict[str, object]:
    """Parse LIN movie into structured data."""
    bids = [b.strip().upper().replace("!", "") for b in LIN_BID_RE.findall(lin)]
    
    # Normalize bids to history tokens: PASS->P, X/D->X, XX/R->XX
    bids_hist = []
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
    
    # Parse hands
    handsraw = LIN_MD_HANDS_RE.findall(lin)
    hands_bbo = []
    if handsraw:
        raw = handsraw[0].strip().upper()
        parts = [p for p in raw.strip(',').split(',') if p != ""]
        if len(parts) != 4 or any(('S' not in p) for p in parts):
            if '.' in raw and ',' not in raw:
                dot_parts = [p for p in raw.split('.') if p]
                if len(dot_parts) >= 4:
                    accum = []
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
        # Convert SWNE to NESW order
        if len(parts) == 4:
            parts_swne = parts
            parts = [parts_swne[2], parts_swne[3], parts_swne[0], parts_swne[1]]  # NESW
        hands_bbo = parts if len(parts) == 4 else []

    return {
        "hands_bbo": hands_bbo,  # NESW order
        "dealer": dealer,
        "vuln": vuln,
        "auction": bids_hist,
        "played_cards": cards,
    }

def bbo_to_dotted(hand_bbo: str) -> str:
    """Convert BBO hand format to dotted notation."""
    if not hand_bbo:
        return ""
    try:
        s_idx = hand_bbo.index("S")
        h_idx = hand_bbo.index("H") 
        d_idx = hand_bbo.index("D")
        c_idx = hand_bbo.index("C")
    except ValueError:
        return ""
    
    sp = hand_bbo[s_idx + 1 : h_idx]
    he = hand_bbo[h_idx + 1 : d_idx]
    di = hand_bbo[d_idx + 1 : c_idx]
    cl = hand_bbo[c_idx + 1 : ]
    return f"{sp}.{he}.{di}.{cl}"

# Bridge game logic helpers
def _dealer_index(seat: str) -> int:
    """Get index of dealer seat."""
    try:
        return SEATS.index(seat.upper())
    except:
        return 0

def _is_contract_bid(tok: str) -> bool:
    """Check if token is a contract bid."""
    t = tok.upper()
    return len(t) >= 2 and t[0] in "1234567" and t[1] in "CDHSN"

def _is_auction_token(tok: str) -> bool:
    """Check if token is an auction action."""
    t = tok.upper()
    return _is_contract_bid(t) or t in ("P", "X", "XX")

def _auction_complete(bids: List[str]) -> bool:
    """Check if auction is complete (3 consecutive passes after at least one bid)."""
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

def _compute_declarer(dealer: str, bids: List[str]) -> str:
    """Compute the declarer from bidding sequence."""
    d_idx = _dealer_index(dealer)
    final_idx = None
    final_strain = None
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

def _left_of(seat: str) -> str:
    """Get seat to the left."""
    idx = _dealer_index(seat)
    return SEATS[(idx + 1) % 4]

def _partner_of(seat: str) -> str:
    """Get partner seat."""
    idx = _dealer_index(seat)
    return SEATS[(idx + 2) % 4]

def _final_contract_strain(bids: List[str]) -> Optional[str]:
    """Get the strain of the final contract."""
    for tok in reversed(bids):
        if _is_contract_bid(tok):
            return tok[1].upper()
    return None

def _hand_has_card(hand_bbo: str, card: str) -> bool:
    """Check if hand contains the specified card."""
    if not hand_bbo or len(card) < 2:
        return False
    suit = card[0].upper()
    rank = card[1].upper()
    try:
        s_idx = hand_bbo.index('S')
        h_idx = hand_bbo.index('H')
        d_idx = hand_bbo.index('D')
        c_idx = hand_bbo.index('C')
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
    """Find which seat has the specified card."""
    if len(hands_bbo) != 4:
        return None
    for seat, hand in zip(SEATS, hands_bbo):
        if _hand_has_card(hand, card):
            return seat
    return None

def _trick_winner(trick: List[str], leader: str, trump: Optional[str]) -> str:
    """Determine winner of a trick."""
    led_suit = trick[0][0].upper()
    leader_idx = _dealer_index(leader)
    candidates = []
    
    if trump and trump != 'N' and any(c[0].upper() == trump for c in trick):
        for i, c in enumerate(trick):
            if c[0].upper() == trump:
                candidates.append((i, c))
    else:
        for i, c in enumerate(trick):
            if c[0].upper() == led_suit:
                candidates.append((i, c))
    
    win_rel, win_card = max(candidates, key=lambda x: RANK_VALUES.get(x[1][1].upper(), 0))
    return SEATS[(leader_idx + win_rel) % 4]

def _compute_seat_to_act(dealer: str, hands_bbo: List[str], history_tokens: List[str]) -> str:
    """Compute which seat should act next."""
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

def _map_history_tokens_for_api(family: str, history_tokens: List[str]) -> List[str]:
    """Map tokens for API call."""
    mapped = []
    for t in history_tokens:
        tu = t.upper()
        if tu == 'D':
            mapped.append('X')
        elif tu == 'R':
            mapped.append('XX')
        else:
            mapped.append(tu)
    return mapped

def _bot_family_from_endpoint(endpoint: BotEndpoint) -> str:
    """Get bot family from endpoint URL."""
    url = (endpoint.base_url or "").lower()
    if "/argine/" in url:
        return "argine"
    if "ben.dev.cl.bridgebase.com" in url:
        return "ben"
    return "gib"

def load_human_data(detail_path: str, registrants_path: str) -> pd.DataFrame:
    """Load and process human player data."""
    print(f"Loading human detail data from {detail_path}")
    detail = pd.read_csv(detail_path, sep="\t", dtype=str)
    
    print(f"Loading registrants data from {registrants_path}")
    registrants = pd.read_csv(registrants_path, sep="\t", dtype=str)
    
    # Map team_id to player_name
    team_to_player = dict(zip(registrants["team_id"], registrants["player"]))
    
    # Process detail data
    detail["player_name"] = detail["team_id"].map(team_to_player)
    detail["board_number"] = pd.to_numeric(detail["board_number"], errors="coerce")
    detail["instance_id"] = pd.to_numeric(detail["instance_id"], errors="coerce")
    detail["rawscore"] = pd.to_numeric(detail["rawscore"], errors="coerce")
    detail["final_score"] = pd.to_numeric(detail["final_score"], errors="coerce")
    
    return detail[["tourney_id", "player_name", "board_number", "instance_id", 
                   "result", "rawscore", "final_score", "movie"]].dropna(
                   subset=["tourney_id", "board_number", "instance_id", "movie"])

def load_bot_data(bot_files: List[str]) -> pd.DataFrame:
    """Load and process bot data from multiple files."""
    all_bot_data = []
    
    for bot_file in bot_files:
        if not os.path.exists(bot_file):
            print(f"Warning: Bot file {bot_file} not found, skipping...")
            continue
            
        print(f"Loading bot data from {bot_file}")
        
        # Extract bot name from filename
        bot_name = os.path.basename(bot_file).replace("louis_20250701_20250710_", "").replace("_merged", "")
        
        bot_rows = []
        with open(bot_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line.startswith("RESULT"):
                i += 1
                continue
                
            # Parse RESULT line
            result_data = {}
            for token in line.split():
                if token == "RESULT":
                    continue
                if "=" in token:
                    key, value = token.split("=", 1)
                    result_data[key.strip()] = value.strip()
            
            # Find corresponding MOVIE line
            movie = None
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line.startswith("MOVIE"):
                    movie = next_line[len("MOVIE"):].strip()
                    break
                elif next_line.startswith("RESULT"):
                    break
                j += 1
            
            if movie and result_data.get("tourney") and result_data.get("board") and result_data.get("instance"):
                bot_rows.append({
                    "bot_name": bot_name,
                    "tourney_id": result_data["tourney"],
                    "board_number": int(result_data["board"]),
                    "instance_id": int(result_data["instance"]),
                    "result": result_data.get("result", ""),
                    "rawscore": float(result_data.get("rawscore", 0)),
                    "movie": movie
                })
            
            i = j if j > i else i + 1
        
        all_bot_data.extend(bot_rows)
    
    return pd.DataFrame(all_bot_data)

def calculate_human_mp_scores(human_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MP% for each human player across all their games."""
    print("Calculating human MP% scores...")
    
    # Calculate MP% for each deal
    def calculate_deal_mp(group):
        group = group.copy()
        group = group.sort_values("final_score", ascending=False)
        n = len(group)
        group["rank"] = range(1, n + 1)
        group["mp_pct"] = ((n - group["rank"]) / (n - 1) * 100) if n > 1 else [50.0] * n
        return group
    
    # Group by tournament/board/instance and calculate MP%
    human_df = human_df.groupby(["tourney_id", "board_number", "instance_id"]).apply(calculate_deal_mp).reset_index(drop=True)
    
    # Calculate overall MP% average per player
    player_stats = human_df.groupby("player_name").agg({
        "mp_pct": "mean",
        "tourney_id": "count"  # number of games
    }).rename(columns={"tourney_id": "game_count"}).reset_index()
    
    return human_df, player_stats

def find_best_human_for_instance(human_df: pd.DataFrame, player_stats: pd.DataFrame, 
                                tourney_id: str, board_number: int, instance_id: int, 
                                min_games: int = 100) -> Optional[Tuple[str, str]]:
    """Find the best human player (highest overall MP%) who played this specific instance."""
    
    # Find humans who played this exact instance
    instance_players = human_df[
        (human_df["tourney_id"] == tourney_id) & 
        (human_df["board_number"] == board_number) & 
        (human_df["instance_id"] == instance_id)
    ]["player_name"].unique()
    
    if len(instance_players) == 0:
        return None
    
    # Filter by players with enough games
    qualified_players = player_stats[
        (player_stats["player_name"].isin(instance_players)) & 
        (player_stats["game_count"] >= min_games)
    ]
    
    if len(qualified_players) == 0:
        # Fallback to any player who played this instance
        qualified_players = player_stats[player_stats["player_name"].isin(instance_players)]
        
    if len(qualified_players) == 0:
        return None
    
    # Select player with highest MP%
    best_player = qualified_players.loc[qualified_players["mp_pct"].idxmax()]
    best_player_name = best_player["player_name"]
    
    # Get their movie for this instance
    player_movie = human_df[
        (human_df["tourney_id"] == tourney_id) & 
        (human_df["board_number"] == board_number) & 
        (human_df["instance_id"] == instance_id) & 
        (human_df["player_name"] == best_player_name)
    ]["movie"].iloc[0]
    
    return best_player_name, player_movie

def call_bot_api(endpoint: BotEndpoint, dealer: str, vuln: str, hands_bbo: List[str], 
                history: List[str], timeout: int = 30, debug: bool = False) -> Tuple[str, Optional[str]]:
    """Call bot API and return predicted action."""
    
    try:
        # Compute who should act next
        pov_final = _compute_seat_to_act(dealer, hands_bbo, history)
        bids_only = [t for t in history if _is_auction_token(t)]
        family = _bot_family_from_endpoint(endpoint)
        
        # Handle GIB declarer/dummy switching
        if _auction_complete(bids_only):
            declarer = _compute_declarer(dealer, bids_only)
            if declarer and family in ("gib") and pov_final == _partner_of(declarer):
                pov_final = declarer

        # Convert hands to dotted format
        if len(hands_bbo) != 4 or any(not h for h in hands_bbo):
            raise ValueError(f"Invalid hands for API call: {hands_bbo}")
        
        hands_dotted = [bbo_to_dotted(h).upper() for h in hands_bbo]
        n, e, s, w = hands_dotted[0], hands_dotted[1], hands_dotted[2], hands_dotted[3]

        # Map history tokens for API
        h_tokens_api = _map_history_tokens_for_api(family, history)
        h_str = "-".join(h_tokens_api)

        # Prepare API parameters
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

        # Make request with retries
        prepared = requests.Request("GET", endpoint.base_url, params=params).prepare()
        
        if debug:
            print(f"    API Call: {prepared.url}")
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                with requests.Session() as session:
                    resp = session.send(prepared, timeout=timeout)
                resp.raise_for_status()
                text = resp.text
                
                if debug:
                    print(f"    API Response: {text}")
                
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

        # Parse response
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
        elif rtype == "play":
            card = (r.get("card") or r.get("pc") or "").upper()
            return ("PLAY", card)
        elif rtype == "result" or rtype == "claim":
            return ("RESULT", None)
        
        # Fallback parsing
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
        
    except Exception as e:
        if debug:
            print(f"    API Error: {e}")
        return ("ERROR", None)

def calculate_deviations_with_api(bot_movie: str, player_movie: str, 
                                endpoint: BotEndpoint, tourney_id: str, board_number: int, 
                                instance_id: int, bot_name: str, player_name: str,
                                timeout: int = 30, debug: bool = False) -> Dict[str, int]:
    """Calculate deviations by calling bot API only when it's the target player's turn to act."""
    
    deviations = {
        "total_deviations": 0,
        "bidding_deviations": 0,
        "opening_lead_deviations": 0,
        "play_deviations": 0,
        "step_details": []
    }
    
    # Parse both movies
    try:
        player_parsed = parse_lin_core(player_movie)
        bot_parsed = parse_lin_core(bot_movie)  # We may not need this, but useful for validation
    except Exception as e:
        if debug:
            print(f"  Failed to parse movies: {e}")
        return deviations
    
    # Extract game information from player movie
    hands_bbo = player_parsed.get("hands_bbo", [])
    dealer = player_parsed.get("dealer", "N")
    vuln = player_parsed.get("vuln", "-")
    
    if not hands_bbo or len(hands_bbo) != 4:
        if debug:
            print(f"  No valid hands found in player movie")
        return deviations
    
    # Create token sequence from player movie (auction + play)
    auction_tokens = player_parsed.get("auction", [])
    play_tokens = player_parsed.get("played_cards", [])
    player_tokens = auction_tokens + play_tokens
    
    if not player_tokens:
        if debug:
            print(f"  No tokens found in player movie")
        return deviations
    
    # Player is always South, partner is always North
    target_player_seat = 'S'  # Player is always South
    partner_seat = 'N'  # Partner is always North
    
    if debug:
        print(f"  Target player is South, partner is North")
    
    # Track game state
    bidding_complete = False
    opening_lead_played = False
    
    # Step-by-step comparison with API calls ONLY for target player's turns
    for step in range(len(player_tokens)):
        # History up to this step uses actual player choices (enforcing constraint)
        history_so_far = player_tokens[:step]
        
        # Determine whose turn it is to act at this step
        seat_to_act = _compute_seat_to_act(dealer, hands_bbo, history_so_far)
        
        # Actual player action at this step
        actual_player_action = player_tokens[step]
        
        # Only call API if it's the target player's turn (South) or partner's turn (North as dummy)
        if seat_to_act == target_player_seat or seat_to_act == partner_seat:
            # Get bot prediction for this step
            try:
                action_type, bot_prediction = call_bot_api(
                    endpoint, dealer, vuln, hands_bbo, history_so_far, timeout, debug=debug
                )
            except Exception as e:
                if debug:
                    print(f"  API call failed at step {step+1}: {e}")
                bot_prediction = ""
                action_type = "ERROR"
            
            # Compare prediction vs actual
            deviation_occurred = (bot_prediction != actual_player_action) if bot_prediction else True
            
            if deviation_occurred:
                deviations["total_deviations"] += 1
                
                # Categorize the deviation by game phase
                if not bidding_complete and step < len(auction_tokens):
                    # Still in bidding phase
                    deviations["bidding_deviations"] += 1
                elif not opening_lead_played:
                    # Opening lead
                    deviations["opening_lead_deviations"] += 1
                    opening_lead_played = True
                else:
                    # Subsequent play
                    deviations["play_deviations"] += 1
            
            # Debug output for target player's steps
            if debug:
                phase = "bidding" if not bidding_complete else ("opening" if not opening_lead_played else "play")
                dev_str = "DEVIATION" if deviation_occurred else "match"
                print(f"    Step {step+1} ({phase}): Bot predicted '{bot_prediction}', "
                      f"player played '{actual_player_action}' -> {dev_str}")
        else:
            # This is an opponent's move - don't call API, just record it
            bot_prediction = ""
            action_type = "OPPONENT"
            deviation_occurred = False
            
            if debug:
                phase = "bidding" if not bidding_complete else ("opening" if not opening_lead_played else "play")
                print(f"    Step {step+1} ({phase}): Opponent ({seat_to_act}) played '{actual_player_action}' -> skipped (not South/North)")
        
        # Check if bidding is complete after this action
        if not bidding_complete and step < len(auction_tokens):
            if _auction_complete(history_so_far + [actual_player_action]):
                bidding_complete = True
        elif not opening_lead_played and bidding_complete:
            opening_lead_played = True
        
        # Store step details for debugging
        step_detail = {
            "step": step + 1,
            "seat_to_act": seat_to_act,
            "target_player_seat": target_player_seat,
            "is_target_player_turn": seat_to_act in [target_player_seat, partner_seat],
            "bot_prediction": bot_prediction or "",
            "player_actual": actual_player_action,
            "deviation": deviation_occurred,
            "phase": "bidding" if not bidding_complete else ("opening" if not opening_lead_played else "play"),
            "action_type": action_type
        }
        deviations["step_details"].append(step_detail)
    
    return deviations

# --- Vulnerability helpers ---
_DEF_V_SCHEDULE = [0,1,2,3, 1,2,3,0, 2,3,0,1, 3,0,1,2]

def _extract_sv_code_from_movie(movie: Optional[str]) -> Optional[int]:
    """Extract vulnerability code from LIN movie string."""
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
    """Convert vulnerability letter to numeric code."""
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
    """Get vulnerability from board number using standard schedule."""
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
    """Add proper vulnerability parameter to endpoint based on bot type."""
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

def load_skill_bucket_profiles(profiles_path: str) -> pd.DataFrame:
    """Load skill bucket profiles from parquet file."""
    try:
        return pd.read_parquet(profiles_path)
    except Exception as e:
        print(f"Warning: Could not load skill profiles from {profiles_path}: {e}")
        return pd.DataFrame()

def assign_skill_buckets_to_players(player_stats: pd.DataFrame, skill_profiles: pd.DataFrame) -> pd.DataFrame:
    """Assign skill buckets to players based on their MP% scores."""
    if skill_profiles.empty:
        # Default assignment if no profiles available
        player_stats = player_stats.copy()
        player_stats["skill_bucket"] = "Unknown"
        return player_stats
    
    # Sort profiles by average MP%
    sorted_profiles = skill_profiles.sort_values("avg_mp_pct")
    
    # Create player stats copy with skill buckets
    player_stats = player_stats.copy()
    player_stats["skill_bucket"] = "Unknown"
    
    for _, player in player_stats.iterrows():
        player_mp = player["mp_pct"]
        
        # Find the closest skill bucket by MP%
        distances = abs(sorted_profiles["avg_mp_pct"] - player_mp)
        closest_bucket = sorted_profiles.loc[distances.idxmin(), "skill_bucket"]
        player_stats.loc[player_stats["player_name"] == player["player_name"], "skill_bucket"] = closest_bucket
    
    return player_stats

def find_player_for_instance_by_skill(human_df: pd.DataFrame, player_stats: pd.DataFrame, 
                                    tourney_id: str, board_number: int, instance_id: int, 
                                    target_skill_bucket: str, min_games: int = 100) -> Optional[Tuple[str, str]]:
    """Find a player from the specified skill bucket who played this instance."""
    
    # Find humans who played this exact instance
    instance_players = human_df[
        (human_df["tourney_id"] == tourney_id) & 
        (human_df["board_number"] == board_number) & 
        (human_df["instance_id"] == instance_id)
    ]["player_name"].unique()
    
    if len(instance_players) == 0:
        return None
    
    # Filter by players with enough games
    qualified_players = player_stats[
        (player_stats["player_name"].isin(instance_players)) & 
        (player_stats["game_count"] >= min_games)
    ]
    
    if len(qualified_players) == 0:
        # Fallback to any player who played this instance
        qualified_players = player_stats[player_stats["player_name"].isin(instance_players)]
        
    if len(qualified_players) == 0:
        return None
    
    # Filter by target skill bucket
    skill_bucket_players = qualified_players[qualified_players["skill_bucket"] == target_skill_bucket]
    
    if len(skill_bucket_players) == 0:
        # No players from target skill bucket, fallback to best available player
        best_player = qualified_players.loc[qualified_players["mp_pct"].idxmax()]
    else:
        # From skill bucket players, pick the one closest to the bucket's average MP%
        # For now, just pick the best one from that bucket
        best_player = skill_bucket_players.loc[skill_bucket_players["mp_pct"].idxmax()]
    
    best_player_name = best_player["player_name"]
    
    # Get their movie for this instance
    player_movie = human_df[
        (human_df["tourney_id"] == tourney_id) & 
        (human_df["board_number"] == board_number) & 
        (human_df["instance_id"] == instance_id) & 
        (human_df["player_name"] == best_player_name)
    ]["movie"].iloc[0]
    
    return best_player_name, player_movie

def main():
    parser = argparse.ArgumentParser(description="Shadow player analysis for dataset_analysis files")
    parser.add_argument("--human-detail", default="detail_20250701_20250710_merged.tsv", 
                       help="Human detail TSV file")
    parser.add_argument("--registrants", default="registrants_20250701_20250710_merged.tsv", 
                       help="Registrants TSV file")
    parser.add_argument("--bot-files", nargs="+", 
                       default=[
                           "louis_20250701_20250710_argine_merged",
                           "louis_20250701_20250710_ben_advanced_merged",
                           "louis_20250701_20250710_ben_beginner_merged",
                           "louis_20250701_20250710_ben_intermediate_merged",
                           "louis_20250701_20250710_ben_novice_merged",
                           "louis_20250701_20250710_gib_advanced_merged",
                           "louis_20250701_20250710_gib_basic_merged"
                       ], help="Bot replay files")
    parser.add_argument("--output", default="shadow_analysis.parquet", help="Output parquet file")
    parser.add_argument("--min-games", type=int, default=100, 
                       help="Minimum games for human player to be considered")
    parser.add_argument("--limit", type=int, help="Limit number of comparisons per bot file for testing")
    parser.add_argument("--timeout", type=int, default=30, help="API timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--player-level", choices=["Beginner", "Novice", "Intermediate", "Advanced"], 
                       help="Target player skill level (if not specified, uses best player)")
    parser.add_argument("--skill-profiles", default="bridge_output_run/skill_bucket_profiles.parquet",
                       help="Skill bucket profiles parquet file")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading human data...")
    human_df = load_human_data(args.human_detail, args.registrants)
    
    print("Calculating human MP% scores...")
    human_df, player_stats = calculate_human_mp_scores(human_df)
    
    # Load skill bucket profiles if player_level is specified
    skill_profiles = pd.DataFrame()
    if args.player_level:
        print(f"Loading skill bucket profiles from {args.skill_profiles}...")
        skill_profiles = load_skill_bucket_profiles(args.skill_profiles)
        if not skill_profiles.empty:
            print("Assigning skill buckets to players...")
            player_stats = assign_skill_buckets_to_players(player_stats, skill_profiles)
            print(f"Player skill bucket distribution:")
            bucket_counts = player_stats["skill_bucket"].value_counts()
            for bucket, count in bucket_counts.items():
                print(f"  {bucket}: {count} players")
        else:
            print("Warning: Could not load skill profiles, falling back to best player selection")
    
    print("Loading bot data...")
    bot_df = load_bot_data(args.bot_files)
    
    print(f"Loaded {len(human_df)} human games and {len(bot_df)} bot games")
    print(f"Found {len(player_stats)} human players")
    
    # Load skill profiles if specified
    skill_profiles = pd.DataFrame()
    if args.skill_profiles:
        print(f"Loading skill bucket profiles from {args.skill_profiles}...")
        skill_profiles = load_skill_bucket_profiles(args.skill_profiles)
        player_stats = assign_skill_buckets_to_players(player_stats, skill_profiles)
    
    # Results storage
    results = []
    
    # Process each bot separately to apply limit per bot file
    total_processed = 0
    for bot_name in bot_df["bot_name"].unique():
        bot_subset = bot_df[bot_df["bot_name"] == bot_name]
        processed_per_bot = 0
        
        print(f"Processing bot: {bot_name} ({len(bot_subset)} games)")
        
        for _, bot_row in bot_subset.iterrows():
            if args.limit and processed_per_bot >= args.limit:
                print(f"  Reached limit of {args.limit} for bot {bot_name}")
                break
                
            tourney_id = bot_row["tourney_id"]
            board_number = bot_row["board_number"] 
            instance_id = bot_row["instance_id"]
            bot_movie = bot_row["movie"]
            
            # Find best human player for this instance
            if args.player_level and not skill_profiles.empty:
                # Find player by skill bucket
                best_match = find_player_for_instance_by_skill(
                    human_df, player_stats, tourney_id, board_number, instance_id, 
                    args.player_level, args.min_games
                )
                selection_method = f"skill_bucket_{args.player_level}"
            else:
                # Find best player regardless of skill bucket
                best_match = find_best_human_for_instance(
                    human_df, player_stats, tourney_id, board_number, instance_id, args.min_games
                )
                selection_method = "best_player"
            
            if not best_match:
                if args.debug:
                    print(f"No qualified human found for {tourney_id}/{board_number}/{instance_id}")
                continue
                
            best_player, player_movie = best_match
            
            # Parse movies into tokens (for length info, actual parsing done in deviation calc)
            bot_tokens = parse_lin_tokens(bot_movie)
            player_tokens = parse_lin_tokens(player_movie)
            
            if not bot_tokens or not player_tokens:
                if args.debug:
                    print(f"Failed to parse movies for {tourney_id}/{board_number}/{instance_id}")
                continue
            
            # Get bot endpoint
            endpoint = endpoint_for_bot(bot_name)
            if not endpoint:
                if args.debug:
                    print(f"No endpoint found for bot {bot_name}")
                continue
            
            # Adjust endpoint for vulnerability
            endpoint = _shadow_endpoint_with_ben_v(endpoint, None, bot_name, player_movie, board_number)
            
            # Calculate deviations with step-by-step API calls
            deviations = calculate_deviations_with_api(
                bot_movie, player_movie, endpoint, tourney_id, board_number, 
                instance_id, bot_name, best_player, timeout=args.timeout, debug=args.debug
            )
            
            # Store result with enhanced debugging info
            player_mp_pct = player_stats[player_stats["player_name"] == best_player]["mp_pct"].iloc[0]
            player_skill_bucket = player_stats[player_stats["player_name"] == best_player]["skill_bucket"].iloc[0] if "skill_bucket" in player_stats.columns else "Unknown"
            
            result = {
                "tourney_id": tourney_id,
                "board_number": board_number,
                "instance_id": instance_id,
                "bot_name": bot_name,
                "best_human_player": best_player,
                "best_human_mp_pct": player_mp_pct,
                "player_skill_bucket": player_skill_bucket,
                "selection_method": selection_method,
                "total_deviations": deviations["total_deviations"],
                "bidding_deviations": deviations["bidding_deviations"], 
                "opening_lead_deviations": deviations["opening_lead_deviations"],
                "play_deviations": deviations["play_deviations"],
                "bot_movie_length": len(bot_tokens),
                "player_movie_length": len(player_tokens),
                "comparison_length": min(len(bot_tokens), len(player_tokens)),
                # Store detailed step information as JSON string for parquet compatibility
                "step_details_json": str(deviations.get("step_details", []))
            }
            
            results.append(result)
            processed_per_bot += 1
            total_processed += 1
            
            if args.debug:
                print(f"DEBUG: Tournament {tourney_id}, Board {board_number}, Instance {instance_id}")
                print(f"  Bot: {bot_name} vs Human: {best_player} (MP%: {player_mp_pct:.1f}, Skill: {player_skill_bucket})")
                print(f"  Selection method: {selection_method}")
                print(f"  Total deviations: {deviations['total_deviations']}")
                print(f"    Bidding: {deviations['bidding_deviations']}")
                print(f"    Opening Lead: {deviations['opening_lead_deviations']}")  
                print(f"    Play: {deviations['play_deviations']}")
                print("  Step-by-step breakdown:")
                for detail in deviations.get("step_details", [])[:10]:  # Show first 10 steps
                    dev_marker = "❌" if detail["deviation"] else "✅"
                    print(f"    {dev_marker} Step {detail['step']} ({detail['phase']}): "
                          f"Bot '{detail['bot_prediction']}' vs Player '{detail['player_actual']}'")
                if len(deviations.get("step_details", [])) > 10:
                    print(f"    ... and {len(deviations['step_details']) - 10} more steps")
                print()
            
            if total_processed % 100 == 0:
                print(f"Processed {total_processed} total comparisons...")
        
        print(f"  Completed {processed_per_bot} comparisons for bot {bot_name}")
        print()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_parquet(args.output, index=False)
        print(f"Saved {len(results)} shadow analysis results to {args.output}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total comparisons: {len(results)}")
        print(f"Average total deviations per game: {results_df['total_deviations'].mean():.2f}")
        print(f"Average bidding deviations: {results_df['bidding_deviations'].mean():.2f}")
        print(f"Average play deviations: {results_df['play_deviations'].mean():.2f}")
        print(f"Bots analyzed: {sorted(results_df['bot_name'].unique())}")
        
        # Per-bot summary
        print(f"\nPer-bot results:")
        for bot_name in sorted(results_df['bot_name'].unique()):
            bot_results = results_df[results_df['bot_name'] == bot_name]
            print(f"  {bot_name}: {len(bot_results)} comparisons, "
                  f"avg deviations: {bot_results['total_deviations'].mean():.2f}")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()
