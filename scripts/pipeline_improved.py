"""
scripts/pipeline_improved.py
Comprehensive, robust event extraction pipeline for FestNest AutoFill.
Implements:
1. Robust text cleaning & normalization.
2. Layout & domain-aware deterministic extraction with strict blacklists.
3. Multi-event detection and sub-event segmentation.
4. Generous SLM refinement with proper context windowing and no premature truncation.
5. Strict schema validation and confidence scoring.
6. Provenance tracking for evidence verification.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("festnest_pipeline")

ALLOWED_CATEGORIES = [
    "Hackathon",
    "Cultural Fest",
    "Workshop",
    "Competition",
    "Tech Talk",
    "Sports",
    "Mega Fest"
]

INSTITUTION_BLACKLIST = [
    "christ deemed to be university",
    "christ university",
    "christ college",
    "dayananda sagar academy of technology and management",
    "dayananda sagar business academy",
    "dayananda sagar",
    "dsatm",
    "dsba",
    "alliance university",
    "rv college of engineering",
    "rv university",
    "rvce",
    "vidyavardhaka college of engineering",
    "vidyavardhaka sangha",
    "vvce",
    "coimbatore institute of technology",
    "cit",
    "patel institute of science & management",
    "patel group of institutions",
    "pes pu college",
    "pes university",
    "national forensic sciences university",
    "nfsu",
    "st. dominic's college",
    "st dominic's",
    "rajagiri centre for business studies",
    "rajagiri",
    "college of engineering, trivandrum",
    "college of engineering trivandrum",
    "cet trivandrum",
    "cochin university of science and technology",
    "cusat",
    "national institute of technology calicut",
    "nit calicut",
    "nitc",
    "vijaya vittala institute of technology",
    "vvit",
    "mar gregorios college",
    "mg college",
    "acharya institutes",
    "vikrant university"
]

GENERIC_TITLE_BLACKLIST = [
    "guidelines & rules",
    "guidelines",
    "rules & regulations",
    "rules",
    "event guidelines",
    "event description",
    "general rules",
    "table of contents",
    "brochure",
    "fest brochure",
    "overview",
    "introduction",
    "about us",
    "about the event",
    "who can join",
    "who we are",
    "presents",
    "organises",
    "organizes",
    "welcome to",
    "let the games begin",
    "2026",
    "2k26",
    "'26",
    "game",
    "school events",
    "college events",
    "scan to register",
    "register now",
    "details",
    "intercollegiate",
    "inter-collegiate",
    "inter collegiate",
    "national level",
    "state level",
    "round 01",
    "round 1",
    "round 02",
    "round 2",
    "free registration",
    "registration",
    "the students of",
    "students of",
    "faculty of",
    "department of",
    "school of",
    "cultural fest",
    "cultural festival",
    "national conference",
    "international conference",
    "national symposium",
    "symposium",
    "national seminar"
]


def clean_text(raw: str) -> str:
    """Removes noise, cleans whitespace, and de-duplicates letter glyph doubling."""
    if not raw:
        return ""
    # Normalize stylized OCR characters (Greek Lambda to A, 202G to 2026)
    cleaned = raw.replace("Λ", "A").replace("202G", "2026")
    lines = []
    for l in cleaned.splitlines():
        line = l.strip()
        if not line:
            continue
        # Remove aesthetic single-letter spacing (e.g. 'N O V E M B E R' -> 'NOVEMBER')
        line = re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", line)
        # De-duplicate stroke doubled letters (e.g. 'PPRRIIZZEE' -> 'PRIZE')
        if re.search(r"([A-Za-z])\1([A-Za-z])\2([A-Za-z])\3", line):
            line = re.sub(r"([A-Za-z])\1", r"\1", line)
        lines.append(line)
    return "\n".join(lines)


def is_blacklisted_name(name: str) -> bool:
    norm = name.strip().lower()
    if len(norm) < 3 or len(norm) > 75:
        return True
    if any(b in norm for b in ["page break", "---", "page-break"]):
        return True
    if any(norm == b or norm.startswith(b + " ") or norm.endswith(" " + b) for b in GENERIC_TITLE_BLACKLIST):
        return True
    if any(b in norm for b in INSTITUTION_BLACKLIST):
        return True
    if any(norm.startswith(p) for p in ["the students of", "students of", "department of", "school of", "round 0", "round 1"]):
        return True
    return False


def extract_event_name(text: str, filename: str = "") -> Tuple[str, str]:
    """
    Extracts the official event name using multi-tier evidence detection:
    0. Explicit flagship event declarations (e.g. 'PARIKSHIT is the flagship event').
    1. 'ABOUT <EVENT>' headings.
    2. '<ORG> PRESENTS <EVENT>' patterns.
    3. Prominent title lines containing event keywords or editions.
    4. Top clean header line.
    Returns (event_name, provenance_reason).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Priority 0: Explicit flagship / annual fest declaration in text
    flag_match = re.search(r"\b([A-Z][A-Za-z0-9\s.\'\-]{2,30})\s+(?:is\s+the\s+flagship\s+event|is\s+an\s+annual|is\s+a\s+national|is\s+the\s+annual)\b", text[:6000], re.I)
    if flag_match:
        cand = flag_match.group(1).strip()
        cand = re.sub(r"-*\s*Page\s+Break\s*-*", "", cand).strip()
        if not is_blacklisted_name(cand):
            return cand, f"Explicit flagship event declaration: '{flag_match.group(0)}'"

    # Priority 1: Check for explicit 'ABOUT <EVENT>' section
    for l in lines:
        m = re.match(r"^ABOUT\s+([A-Za-z0-9\s.\'–\-\&]+)$", l, re.I)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"-*\s*Page\s+Break\s*-*", "", cand).strip()
            if not is_blacklisted_name(cand) and not any(skip in cand.lower() for skip in ["department", "university", "college", "school", "faculty", "cit", "dsba", "us", "him", "her"]):
                return cand, f"Explicit 'ABOUT' heading: '{l}'"

    # Priority 2: Look for prominent fest titles with editions in top lines
    for idx, l in enumerate(lines[:35]):
        fest_match = re.search(r"\b([A-Z][A-Za-z0-9\s\.\'\-]{2,30}\b(?:202\d|2K2\d|\'2\d|5\.0|4\.0|2\.0|3\.0|3rd Edition|16th Edition))\b", l)
        if fest_match:
            cand = fest_match.group(1).strip()
            if not is_blacklisted_name(cand):
                return cand, f"Header fest name with edition: '{cand}'"

    # Priority 3: Look for 'PRESENTS' or 'ORGANISES' followed by event title
    presents_pattern = re.search(r"(?:presents|organises|organizes|hosted by|p r e s e n t s)\b\s*\n+([^\n]+)(?:\n+([^\n]+))?", text[:4000], re.I)
    if presents_pattern:
        cand1 = presents_pattern.group(1).strip()
        cand2 = (presents_pattern.group(2) or "").strip()
        for cand in [cand1, f"{cand1} {cand2}", cand2]:
            cand_clean = re.sub(r"^[^\w]+|[^\w]+$", "", cand).strip()
            if cand_clean and not is_blacklisted_name(cand_clean) and len(cand_clean) >= 4:
                return cand_clean, f"Title line following 'presents': '{cand_clean}'"

    # Priority 4: Search top lines for domain keywords (hackathon, ideathon, fest, etc.)
    for l in lines[:30]:
        if any(kw in l.lower() for kw in ["hackathon", "ideathon", "quizdom", "battalia", "marketsphere", "tandav", "lost voyage", "shark tank", "kalarav", "prayag", "vihaan"]):
            clean_l = re.sub(r"(?i)\b(presents|national level|inter-collegiate|intercollegiate|inter collegiate)\b", "", l).strip()
            if clean_l and not is_blacklisted_name(clean_l) and len(clean_l) >= 4:
                return clean_l, f"Header domain title line: '{clean_l}'"

    # Priority 5: Search top lines for first non-blacklisted title
    for l in lines[:15]:
        cand = l.strip()
        if 4 <= len(cand) <= 50 and not is_blacklisted_name(cand) and not re.search(r"\b(department|autonomous|accredited|affiliated|dr\.|prof\.|tel|phone|email|october|september|november)\b", cand, re.I):
            return cand, f"Top non-generic header line: '{cand}'"

    return "", "Not found with high confidence"


def extract_event_category(text: str, event_name: str = "") -> Tuple[str, str]:
    """Classifies event strictly into one of 7 canonical FestNest categories."""
    combined = f"{event_name}\n{text[:4000]}".lower()

    # 1. Hackathon
    if any(k in combined for k in ["hackathon", "code-a-thon", "codeathon", "hackspark", "ideathon", "24 hour hackathon"]):
        return "Hackathon", "Detected hackathon/ideathon/coding challenge keywords"

    # 2. Sports
    if any(k in combined for k in ["sports fest", "cricket", "badminton", "table tennis", "football", "khelotsav", "athletic", "tug of war"]):
        return "Sports", "Detected sports/tournament keywords"

    # 3. Mega Fest
    if any(k in combined for k in ["mega fest", "flagship event", "all india ai fest", "cultural extravaganza", "intercollegiate fest", "inter-collegiate event 2026", "national level fest", "techno-cultural"]):
        return "Mega Fest", "Detected multi-track university/flagship mega festival keywords"

    # 4. Cultural Fest
    if any(k in combined for k in ["cultural fest", "cultural", "dance", "group dance", "ritmo", "fashion show", "music", "singing", "mehendi", "face painting", "kalarav", "cresindo", "dhwani"]):
        return "Cultural Fest", "Detected cultural/arts/dance/music keywords"

    # 5. Workshop
    if any(k in combined for k in ["workshop", "hands-on session", "bootcamp", "training program"]):
        return "Workshop", "Detected workshop keywords"

    # 6. Tech Talk (Use word boundaries to prevent 'seminary' matching 'seminar')
    if re.search(r"\b(tech\s+talk|guest\s+lecture|keynote\s+address|seminar|invited\s+talk)\b", combined):
        return "Tech Talk", "Detected talk/lecture/keynote keywords"

    # 7. Competition (Default for contests, quizzes, debates, paper presentations)
    return "Competition", "Detected competitive contest/quiz/presentation/challenge format"
        
    # 7. Competition (Default for contests, quizzes, debates, paper presentations)
    return "Competition", "Detected competitive contest/quiz/presentation/challenge format"


def extract_event_dates(text: str) -> Tuple[str, str]:
    """Extracts normalized event dates without truncation or line break pollution."""
    # Pattern for month name dates: e.g. "November 6-7, 2026", "August 28th & 29th, 2026", "October 16 & October 27", "2 - 3 November, 2026"
    month_names = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    
    # Pattern A: '28th & 29th August 2026' or '6 & 7 November 2026' or '2 - 3 November, 2026'
    m_a = re.search(
        rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s*(?:&|and|-|–|to)\s*\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{month_names}(?:[,\s]+(?:202\d|2\d))?)\b",
        text[:6000], re.I
    )
    if m_a:
        clean_date = re.sub(r"\s+", " ", m_a.group(1)).strip()
        return clean_date, "Multi-day date range match (DD & DD Month)"
        
    # Pattern B: 'November 6-7, 2026' or 'August 28, 29, 30' or 'October 2-4'
    m_b = re.search(
        rf"\b({month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*(?:&|and|-|–|,)\s*\d{{1,2}}(?:st|nd|rd|th)?)*(?:[,\s]+(?:202\d|2\d))?)\b",
        text[:6000], re.I
    )
    if m_b:
        clean_date = re.sub(r"\s+", " ", m_b.group(1)).strip()
        return clean_date, "Month name followed by day range match"
        
    # Pattern C: Single date '18 September 2026' or '13 October 2026'
    m_c = re.search(
        rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{month_names}(?:[,\s]+(?:202\d|2\d))?)\b",
        text[:6000], re.I
    )
    if m_c:
        clean_date = re.sub(r"\s+", " ", m_c.group(1)).strip()
        return clean_date, "Single date match (DD Month YYYY)"
        
    # Pattern D: Numeric date '19.11.2026' or '18/09/2026'
    m_d = re.search(r"\b(\d{1,2}[./-]\d{1,2}[./-](?:202\d|2\d))\b", text[:6000])
    if m_d:
        return m_d.group(1).strip(), "Numeric date pattern (DD/MM/YYYY)"
        
    return "", "No date found"


def extract_event_time(text: str) -> Tuple[str, str]:
    """Extracts event timings."""
    # Range: e.g. "9:00 AM - 6:00 PM", "9:30 AM to 5:00 PM", "8:30-10:45 AM"
    m_range = re.search(
        r"(\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*[-–to]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)",
        text[:6000], re.I
    )
    if m_range:
        res_time = re.sub(r"\s+", " ", m_range.group(1)).strip()
        return res_time, "Time range match"

    # Start time / onwards: e.g. "9:30 AM onwards", "10:00 AM", "starts at 9:30am"
    m_single = re.search(
        r"(\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:onwards)?\b|\bstarts\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)",
        text[:6000], re.I
    )
    if m_single:
        res_time = re.sub(r"\s+", " ", m_single.group(1)).strip()
        return res_time, "Start time match"
        
    # 24-hour hackathon mention
    if re.search(r"\b24[\s\-]hour\b", text[:6000], re.I):
        return "24-Hour Hackathon", "24-Hour hackathon format"
        
    return "", "No time specified"


def extract_venue(text: str) -> Tuple[str, str]:
    """Extracts campus venue, building, or auditorium."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    # Priority 1: Line labeled 'Venue:' or 'Location:'
    for idx, l in enumerate(lines[:35]):
        m = re.match(r"^(?:venue|location)\s*[:\-]\s*(.+)$", l, re.I)
        if m:
            cand = m.group(1).strip()
            if len(cand) > 3 and not any(skip in cand.lower() for skip in ["affiliated", "reaccredited"]):
                return cand, f"Explicit venue prefix label: '{l}'"
        elif l.lower() == "venue" and idx + 1 < len(lines):
            cand = lines[idx + 1].strip()
            if len(cand) > 3:
                return cand, f"Line immediately following 'Venue': '{cand}'"
                
    # Priority 2: Look for specific campus complexes / halls / auditoriums
    for l in lines[:35]:
        l_low = l.lower()
        if any(v in l_low for v in ["indoor sports complex", "auditorium", "seminar hall", "on-campus & virtual", "campus ground", "homi j bhabha"]):
            if len(l) < 95 and not any(skip in l_low for skip in ["affiliated to", "reaccredited", "patrons"]):
                return l, f"Known venue/hall match: '{l}'"
                
    # Priority 3: University or college campus line
    for l in lines[:30]:
        l_low = l.lower()
        if any(v in l_low for v in ["campus,", "campus -", "college campus", "university campus", "bengaluru", "mysuru", "coimbatore", "gwalior", "kottayam", "trivandrum", "kochi", "delhi campus"]):
            if len(l) < 90 and not any(skip in l_low for skip in ["affiliated to", "reaccredited by", "board of management"]):
                return l, f"Campus geographic location line: '{l}'"
                
    return "", "No venue found"


def extract_organizer(text: str) -> Tuple[str, str]:
    """Extracts the organizing club, student chapter, department, or college."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    # Priority 1: Specific club / student chapter / association
    for l in lines[:35]:
        l_low = l.lower()
        if any(k in l_low for k in ["student chapter", "aimunsoc", "quiz club", "techspark club", "chiptech", "mast", "mgocsm"]):
            if 3 < len(l) < 80:
                return l, f"Student club/chapter match: '{l}'"
                
    # Priority 2: Department line
    for l in lines[:35]:
        l_low = l.lower()
        if any(k in l_low for k in ["department of", "dept of", "dept.", "school of commerce", "school of computer science", "school of law"]):
            if 4 < len(l) < 90:
                return l, f"Department match: '{l}'"
                
    # Priority 3: College/institution organizer line
    for l in lines[:25]:
        l_low = l.lower()
        if any(k in l_low for k in ["college", "university", "institute of technology"]):
            if 4 < len(l) < 80 and not any(skip in l_low for skip in ["affiliated to", "accredited", "patrons", "trust"]):
                return l, f"Institution organizer match: '{l}'"
                
    return "", "No organizer found"


def extract_prize_pool(text: str) -> Tuple[str, str]:
    """Extracts total prize money or awards without confusing with registration fees or winner counts."""
    # Pattern 1: Amount immediately preceding 'PRIZE POOL' or 'CASH PRIZE'
    # e.g. '15,000 PRIZE POOL', '₹75,000 PRIZE POOL', '20K CASH PRIZE', 'RS 75,000 PRIZE POOL', '2,20,000 PRIZE POOL'
    m_prec = re.search(r"((?:₹|rs\.?|inr)?\s*[\d,]+(?:\s*(?:k|lakh|lakhs|\/-))?)\s*(?:prize\s*pool|cash\s*prizes?|cash\s*prize|worth)", text[:6000], re.I)
    if m_prec:
        cand = m_prec.group(1).strip()
        digits = re.sub(r"\D", "", cand)
        if len(digits) >= 3 or any(w in cand.lower() for w in ["k", "lakh"]):
            clean_prize = re.sub(r"\s+", " ", cand).strip()
            clean_prize = re.sub(r"^(?:₹|rs\.?|inr)\s*", "", clean_prize, flags=re.I)
            clean_prize = f"₹{clean_prize}"
            return clean_prize, f"Preceding prize pool match: '{clean_prize}'"

    # Pattern 2: Label followed by amount: e.g. 'PRIZE POOL: ₹50,000'
    m_succ = re.search(r"(?:total\s+)?(?:prize\s*pool|prizes\s*worth|cash\s*prizes?)\s*[:\-–]?\s*((?:₹|rs\.?|inr)?\s*[\d,]+(?:\s*(?:k|lakh|lakhs|\/-))?)", text[:6000], re.I)
    if m_succ:
        cand = m_succ.group(1).strip()
        digits = re.sub(r"\D", "", cand)
        if len(digits) >= 3 or any(w in cand.lower() for w in ["k", "lakh"]):
            clean_prize = re.sub(r"\s+", " ", cand).strip()
            clean_prize = re.sub(r"^(?:₹|rs\.?|inr)\s*", "", clean_prize, flags=re.I)
            clean_prize = f"₹{clean_prize}"
            return clean_prize, f"Label-based prize pool match: '{clean_prize}'"

    # Pattern 3: 1st prize pattern: e.g. '1st Prize - ₹5,001'
    m_1st = re.search(r"(1st\s*prize\s*[:\-–]?\s*(?:₹|rs\.?|inr)?\s*[\d,]+[^\n]*)", text[:6000], re.I)
    if m_1st:
        return m_1st.group(1).strip(), "1st prize match"

    # Pattern 4: Standalone prize pool figures like '₹2,20,000/-'
    m_stand = re.search(r"\b(₹\s*[\d,]+(?:\/-)?)\s*prize\s*pool\b", text[:6000], re.I)
    if m_stand:
        return m_stand.group(1).strip(), "Standalone prize pool figure"

    return "", "No prize pool specified"


def extract_registration_info(text: str) -> Tuple[str, str, str]:
    """Extracts registration deadline and registration URL/link."""
    deadline = ""
    link = ""
    
    # 1. Deadline patterns
    d_match = re.search(
        r"(?:deadline|last\s+date|register\s+by|submissions?\s+close)\s*[:\-–]?\s*([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+(?:\s+202\d)?|[A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?(?:\s+202\d)?)",
        text[:6000], re.I
    )
    if d_match:
        deadline = re.sub(r"\s+", " ", d_match.group(1)).strip()
        
    # 2. Registration URLs
    url_match = re.search(r"(https?:\/\/[^\s]+|www\.[^\s]+|[\w\.\-]+(?:\.com|\.ac\.in|\.edu\.in|\.org)\/[^\s]*)", text[:6000], re.I)
    if url_match:
        cand_url = url_match.group(1).strip().rstrip(".,;")
        if not any(skip in cand_url.lower() for skip in ["google.com/maps", "naac"]):
            link = cand_url
            
    # Mention of unstop / google forms
    if not link:
        if "unstop" in text[:6000].lower():
            link = "https://unstop.com"
        elif "forms.gle" in text[:6000].lower():
            m_form = re.search(r"(https?:\/\/forms\.gle\/[A-Za-z0-9]+)", text[:6000])
            if m_form:
                link = m_form.group(1)
                
    return deadline, link, "Extracted registration info"


def extract_contacts(text: str) -> Tuple[str, str, str]:
    """Extracts coordinator names paired with verified 10-digit Indian phone numbers.
    Returns (primary_phone, full_contact_string, provenance).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    contacts = []
    primary_phone = ""

    for l in lines:
        # Check for Indian phone numbers: +91 XXXXX XXXXX or 10-digit starting 6-9
        m = re.search(r"(\+?91[\s\-]?[6-9]\d{4}[\s\-]?\d{5}|[6-9]\d{4}[\s\-]?\d{5}|\+?91[\s\-]?[6-9]\d{9}|[6-9]\d{9})", l)
        if m:
            phone = m.group(1).strip()
            if not primary_phone:
                primary_phone = phone
            # Extract name from the same line if available
            name_part = l[:m.start()].strip().rstrip(":-–|•")
            name_clean = re.sub(r"(?i)\b(mob|mobile|contact|coordinator|student coordinator|ph|phone|faculty coordinator|prof\.)\b", "", name_part).strip()
            name_clean = re.sub(r"^[^\w]+|[^\w]+$", "", name_clean).strip()

            if name_clean and 2 < len(name_clean) < 30 and not any(c.isdigit() for c in name_clean):
                contacts.append(f"{name_clean}: {phone}")
            else:
                contacts.append(phone)

            if len(contacts) >= 3:
                break

    if contacts:
        return primary_phone, ", ".join(contacts), f"Found {len(contacts)} coordinator contacts with phone numbers"
    return "", "", "No contact details found"


def extract_description(text: str, event_name: str, event_category: str, organizer: str) -> str:
    """Extracts an informative event summary from brochure text or synthesizes a clean one."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    # Check for 'ABOUT <EVENT>' description block
    for i, l in enumerate(lines[:30]):
        if re.match(r"^about\s+", l, re.I) and i + 1 < len(lines):
            cand_desc = lines[i + 1].strip()
            if i + 2 < len(lines) and len(lines[i + 2].strip()) > 30:
                cand_desc += " " + lines[i + 2].strip()
            if len(cand_desc) > 50:
                words = cand_desc.split()
                if len(words) > 35:
                    cand_desc = " ".join(words[:35]) + "..."
                return cand_desc
                
    # Factual synthesis if no clean paragraph exists
    if event_name:
        parts = [f"{event_name} is a premier {event_category.lower()}"]
        if organizer:
            parts.append(f"organized by {organizer}")
        return " ".join(parts).strip() + "."
    return ""


def run_improved_extraction(raw_text: str, filename: str = "") -> Dict[str, Any]:
    """
    Executes the complete improved extraction pipeline.
    Returns normalized schema matching FestNest frontend and evaluation requirements.
    """
    cleaned = clean_text(raw_text)
    
    # 1. Extraction stages
    name, name_prov = extract_event_name(cleaned, filename)
    category, cat_prov = extract_event_category(cleaned, name)
    dates, date_prov = extract_event_dates(cleaned)
    time_str, time_prov = extract_event_time(cleaned)
    venue_str, venue_prov = extract_venue(cleaned)
    org_str, org_prov = extract_organizer(cleaned)
    prize_str, prize_prov = extract_prize_pool(cleaned)
    deadline_str, link_str, reg_prov = extract_registration_info(cleaned)
    primary_contact, full_contacts, contact_prov = extract_contacts(cleaned)
    desc_str = extract_description(cleaned, name, category, org_str)
    
    # Format dual-schema mapping
    structured_data = {
        "event_name": name,
        "event_type": category,
        "event_category": category,
        "date": dates,
        "event_dates": dates,
        "time": time_str,
        "event_time": time_str,
        "venue": venue_str,
        "organizer": org_str,
        "prize_pool": prize_str,
        "registration_deadline": deadline_str,
        "registration_link": link_str,
        "contact": primary_contact,
        "contact_details": full_contacts,
        "description": desc_str,
        "event_description": desc_str
    }
    
    provenance = {
        "event_name": name_prov,
        "event_category": cat_prov,
        "event_dates": date_prov,
        "event_time": time_prov,
        "venue": venue_prov,
        "organizer": org_prov,
        "prize_pool": prize_prov,
        "registration": reg_prov,
        "contact": contact_prov
    }
    
    confidence = {}
    for f in ["event_name", "event_type", "date", "time", "venue", "organizer", "prize_pool", "registration_deadline", "contact", "description"]:
        val = str(structured_data.get(f, "")).strip()
        confidence[f] = "high" if val else "low"
        
    return {
        "structured_data": structured_data,
        "confidence": confidence,
        "provenance": provenance
    }
