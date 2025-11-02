
import json, re, numpy as np, pandas as pd
from typing import Any, Dict, List, Tuple, Optional

# ---------- helpers ----------
NAME_PAT = re.compile(r"[®™\u00AE\u2122]")

def canon_name(s):
    if not s or not isinstance(s, str):
        return None
    s2 = NAME_PAT.sub("", s)
    s2 = re.sub(r"[^a-zA-Z0-9\-\s\+\/\(\)\.,]", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip().lower()
    return s2

def parse_date(s):
    try:
        return pd.to_datetime(s).tz_localize(None)
    except Exception:
        return None

def to_float_safe(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# ---------- label taxonomy ----------
PRIMARY_LABELS = [
    "Immune","Energy","Sleep/Calm","Cognitive/Focus","Joint/Bone",
    "Heart/Cardio","Digestive/Gut","Men’s/Women’s","Sports/Performance","General Wellness"
]

# Regex patterns as **raw strings** (compiled here in-module)
CLAIM_PATTERNS_RAW = {
    "Immune": [
        r"\bimmun", r"\bdefen[cs]e?", r"\b(resist|resilien)",
        r"\b(cold|flu|season(al)?)\b", r"\bupper respiratory\b",
        r"antioxid(ant|ative|ant activity)", r"\binflamm(ation|atory)\b"
    ],
    "Energy": [
        r"\benerg(y|ize|ising|izing)\b", r"\bfatigue\b", r"\btired(ness)?\b",
        r"\bstamina\b", r"\bvitalit(y|e)\b", r"\bmetaboli[cs]\b", r"\bthermogenic\b"
    ],
    "Sleep/Calm": [
        r"\bsleep\b", r"\bmelatonin\b", r"\b(relax|relaxation)\b",
        r"\bcalm(ing)?\b", r"\bstress\b", r"\banxiet(y|ies)\b",
        r"\brestful\b", r"\bserotonin\b", r"\bcircadian\b"
    ],
    "Cognitive/Focus": [
        r"\bcognit(ion|ive)\b", r"\bmemor(y|ies)\b", r"\bfocus\b",
        r"\battention\b", r"\bclarit(y|y of mind)\b", r"\bbrain\b",
        r"\bnootrop(ic|ic[s]?)\b", r"\bneuro\b", r"\blearning\b"
    ],
    "Joint/Bone": [
        r"\bjoint(s)?\b", r"\bbone(s)?\b", r"\bcartilag(e)?\b",
        r"\bmobilit(y)?\b", r"\bflexib(il|il)it(y)?\b", r"\boste(oa|o)r(th)?",
        r"\bskeleton\b"
    ],
    "Heart/Cardio": [
        r"\bheart\b", r"\bcardio(vascular)?\b", r"\bcholesterol\b",
        r"\btriglycerid(es)?\b", r"\bblood pressure\b", r"\bcirculat(ion|ory)\b",
        r"\bhdl\b", r"\bldl\b"
    ],
    "Digestive/Gut": [
        r"\bdigest(ion|ive)\b", r"\bgut\b", r"\b(pro|pre)biotic(s)?\b",
        r"\bfiber\b", r"\b(en)?zym(e|es)\b", r"\bbowel\b", r"\bregularit(y)?\b",
        r"\bmicrobiome\b", r"\b(bloat|gas|constipat|diarrh)\w*"
    ],
    "Men’s/Women’s": [
        r"\bmen['’]s\b", r"\bmale(s)?\b", r"\btestosteron(e)?\b", r"\bprostate\b",
        r"\bwomen['’]s\b", r"\bfemale(s)?\b", r"\bprenatal\b", r"\bpostnatal\b",
        r"\bpregnan(cy|t)\b", r"\bmenopaus(e|al)\b", r"\bpms\b"
    ],
    "Sports/Performance": [
        r"\bsport(s)?\b", r"\bperformance\b", r"\bendurance\b", r"\bstamina\b",
        r"\bpre[- ]?workout\b", r"\bpost[- ]?workout\b", r"\brecover(y|ies)\b",
        r"\bmuscle\b", r"\bstrength\b", r"\bvo2\b", r"\bnitric oxide\b", r"\bpump\b"
    ],
    "General Wellness": [
        r"\bwellness\b", r"\boverall health\b", r"\bgeneral health\b",
        r"\bdaily support\b", r"\bmultivitamin\b", r"\bmulti[- ]?vitamin\b",
        r"\bdaily vitamin(s)?\b"
    ],
}
CLAIM_PATTERNS = {k:[re.compile(p, re.I) for p in v] for k,v in CLAIM_PATTERNS_RAW.items()}

def map_claims_regex(record, patterns=None, labels=None):
    if patterns is None: patterns = CLAIM_PATTERNS
    if labels is None: labels = PRIMARY_LABELS
    texts = []
    for c in record.get("claims", []) or []:
        texts.append(str(c.get("langualCodeDescription","")))
    for s in record.get("statements", []) or []:
        texts.append(str(s.get("notes","")))
    blob = " || ".join(texts)
    hits = set()
    for lab, regs in patterns.items():
        if any(r.search(blob) for r in regs):
            hits.add(lab)
    return [lab for lab in labels if lab in hits]

# ---------- main export: parse_record ----------
def parse_record(path: str) -> Tuple[Dict[str,Any], List[Dict[str,Any]], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    prod = {
        "path": path,
        "id": r.get("id"),
        "productVersionCode": r.get("productVersionCode"),
        "fullName": r.get("fullName"),
        "brandName": r.get("brandName"),
        "upcSku": r.get("upcSku"),
        "entryDate": parse_date(r.get("entryDate")),
        "offMarket": int(r.get("offMarket") or 0),
        "productType": (r.get("productType") or {}).get("langualCodeDescription"),
        "physicalState": (r.get("physicalState") or {}).get("langualCodeDescription"),
        "servingsPerContainer": r.get("servingsPerContainer"),
    }

    ss = (r.get("servingSizes") or [])
    if ss:
        s0 = ss[0]
        prod["daily_min"] = to_float_safe(s0.get("minDailyServings"))
        prod["daily_max"] = to_float_safe(s0.get("maxDailyServings"))
        prod["serving_unit"] = s0.get("unit")
    else:
        prod["daily_min"] = np.nan
        prod["daily_max"] = np.nan
        prod["serving_unit"] = None

    tgs = r.get("targetGroups") or r.get("userGroups") or []
    if isinstance(tgs, list):
        prod["targetGroups"] = ";".join(sorted({str(x) if isinstance(x,str) else str(x.get('dailyValueTargetGroupName','')) for x in tgs})) or None
    else:
        prod["targetGroups"] = None

    labels = map_claims_regex(r)

    ing_rows = []
    for row in r.get("ingredientRows", []) or []:
        name = canon_name(row.get("name"))
        cat  = row.get("category")
        qlist = row.get("quantity") or []
        for q in qlist:
            ing_rows.append({
                "id": prod["id"],
                "productVersionCode": prod["productVersionCode"],
                "ingredient_name": name,
                "ingredient_category": cat,
                "servingSizeOrder": q.get("servingSizeOrder"),
                "per_serving_qty": to_float_safe(q.get("quantity")),
                "per_serving_unit": q.get("unit"),
                "serving_unit": q.get("servingSizeUnit"),
            })

    return prod, ing_rows, labels
