"""Market-specific algorithms — numerical claim extraction, financial freshness,
ticker linking, and sector-aware source prioritization.

These are lightweight, deterministic algorithms that enhance the Market
Intelligence Workspace without requiring model fine-tuning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ── Numerical Claim Extraction ─────────────────────────────────────────

_FINANCIAL_NUMBER_RE = re.compile(
    r"(?P<value>[\$€£¥]?\s*-?\d[\d,]*\.?\d*)\s*"
    r"(?P<unit>[%BMKTbmkt](?:illion|rillion)?|basis\s+points|bps)?"
    r"(?:\s+(?P<context>revenue|profit|margin|growth|decline|loss|"
    r"earnings|ebitda|eps|p/e|market\s+cap|aum|nav|yield|"
    r"dividend|return|cagr|yoy|qoq|mom))?",
    re.IGNORECASE,
)

_METRIC_KEYWORDS = frozenset({
    "revenue", "profit", "margin", "growth", "earnings", "ebitda",
    "eps", "p/e", "market cap", "aum", "nav", "yield", "dividend",
    "return", "cagr", "yoy", "qoq", "share price", "valuation",
    "debt", "leverage", "cash flow", "operating income", "net income",
})


@dataclass
class NumericalClaim:
    """A financial claim containing specific numerical data."""

    claim_id: str = ""
    text: str = ""
    values: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    is_verifiable: bool = False


def extract_numerical_claims(claims: list[dict[str, Any]]) -> list[NumericalClaim]:
    """Identify claims that contain concrete financial numbers.

    Tags each claim with extracted numeric values and the financial
    metrics they relate to, making them suitable for verification.
    """
    results: list[NumericalClaim] = []
    for claim in claims:
        text = claim.get("text", "")
        matches = list(_FINANCIAL_NUMBER_RE.finditer(text))
        if not matches:
            continue

        values = [m.group("value").strip() for m in matches]
        contexts = [m.group("context") or "" for m in matches]

        lower = text.lower()
        metrics = [kw for kw in _METRIC_KEYWORDS if kw in lower]
        metrics.extend([c for c in contexts if c])
        metrics = list(dict.fromkeys(metrics))

        results.append(NumericalClaim(
            claim_id=claim.get("claim_id", ""),
            text=text[:250],
            values=values,
            metrics=metrics,
            is_verifiable=len(values) > 0 and len(metrics) > 0,
        ))

    return results


def numerical_claim_rate(claims: list[dict[str, Any]]) -> float:
    """Fraction of claims containing financial numbers."""
    if not claims:
        return 0.0
    numerical = extract_numerical_claims(claims)
    return round(len(numerical) / len(claims), 3)


# ── Financial Freshness Scoring ────────────────────────────────────────

_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})\b", re.I),
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s+(\d{4})\b", re.I),
    re.compile(r"\bQ[1-4]\s+(\d{4})\b"),
    re.compile(r"\b(FY|CY)\s*(\d{4})\b", re.I),
]

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class FreshnessScore:
    """Freshness assessment for a financial source."""

    source_id: str = ""
    newest_date: str = ""
    age_days: int = -1
    score: float = 0.0
    date_mentions: int = 0


def score_financial_freshness(
    source_id: str,
    text: str,
    reference_date: datetime | None = None,
) -> FreshnessScore:
    """Score how recent the financial data in a source is.

    Extracts date mentions from the text and computes a recency score
    relative to the reference date (default: now).
    """
    ref = reference_date or datetime.now(UTC)
    dates: list[datetime] = []

    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            parsed = _parse_date_match(m)
            if parsed and parsed.year >= 2015:
                dates.append(parsed)

    if not dates:
        return FreshnessScore(source_id=source_id, score=0.2, date_mentions=0)

    newest = max(dates)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=UTC)
    age_days = max(0, (ref - newest).days)

    if age_days <= 90:
        score = 1.0
    elif age_days <= 365:
        score = 0.8
    elif age_days <= 730:
        score = 0.5
    else:
        score = max(0.1, 0.5 - (age_days - 730) / 3650)

    return FreshnessScore(
        source_id=source_id,
        newest_date=newest.isoformat()[:10],
        age_days=age_days,
        score=round(score, 3),
        date_mentions=len(dates),
    )


def _parse_date_match(m: re.Match) -> datetime | None:
    groups = m.groups()
    try:
        if len(groups) == 3 and groups[0].isdigit() and len(groups[0]) == 4:
            return datetime(int(groups[0]), int(groups[1]), int(groups[2]), tzinfo=UTC)
        if len(groups) >= 3:
            for g in groups:
                month = _MONTH_MAP.get(g[:3].lower())
                if month:
                    year_candidates = [int(x) for x in groups if x.isdigit() and int(x) > 1900]
                    day_candidates = [int(x) for x in groups if x.isdigit() and int(x) <= 31]
                    if year_candidates and day_candidates:
                        return datetime(year_candidates[0], month, day_candidates[0], tzinfo=UTC)
        if len(groups) >= 1:
            year_str = [g for g in groups if g.isdigit() and len(g) == 4]
            if year_str:
                return datetime(int(year_str[0]), 6, 15, tzinfo=UTC)
    except (ValueError, IndexError):
        pass
    return None


# ── Ticker / Company Mention Linking ───────────────────────────────────

_COMMON_TICKERS: dict[str, list[str]] = {
    "AAPL": ["apple", "iphone", "ipad", "macos"],
    "MSFT": ["microsoft", "azure", "windows", "office 365"],
    "GOOGL": ["google", "alphabet", "android", "chrome", "waymo"],
    "AMZN": ["amazon", "aws", "prime"],
    "NVDA": ["nvidia", "geforce", "cuda", "tensorrt"],
    "META": ["meta", "facebook", "instagram", "whatsapp"],
    "TSLA": ["tesla", "model s", "model 3", "model y", "autopilot"],
    "JPM": ["jpmorgan", "jp morgan", "chase"],
    "GS": ["goldman sachs", "goldman"],
    "BRK": ["berkshire", "berkshire hathaway"],
}


@dataclass
class TickerMention:
    """A detected ticker/company reference in text."""

    ticker: str = ""
    company_name: str = ""
    mention_count: int = 0
    positions: list[int] = field(default_factory=list)


def link_ticker_mentions(text: str, target_ticker: str = "") -> list[TickerMention]:
    """Find ticker and company name mentions in source text.

    When *target_ticker* is provided, its mentions are prioritized.
    Returns all detected ticker references sorted by frequency.
    """
    lower = text.lower()
    results: dict[str, TickerMention] = {}

    for ticker, aliases in _COMMON_TICKERS.items():
        positions: list[int] = []

        for m in re.finditer(r"\b" + re.escape(ticker) + r"\b", text):
            positions.append(m.start())

        for alias in aliases:
            for m in re.finditer(re.escape(alias), lower):
                positions.append(m.start())

        if positions:
            results[ticker] = TickerMention(
                ticker=ticker,
                company_name=aliases[0] if aliases else ticker,
                mention_count=len(positions),
                positions=sorted(positions)[:10],
            )

    if target_ticker:
        upper = target_ticker.upper()
        ticker_positions = [m.start() for m in re.finditer(r"\b" + re.escape(upper) + r"\b", text)]
        if ticker_positions and upper not in results:
            results[upper] = TickerMention(
                ticker=upper,
                company_name=upper,
                mention_count=len(ticker_positions),
                positions=ticker_positions[:10],
            )

    mentions = sorted(results.values(), key=lambda t: t.mention_count, reverse=True)
    if target_ticker:
        upper = target_ticker.upper()
        target_first = [m for m in mentions if m.ticker == upper]
        rest = [m for m in mentions if m.ticker != upper]
        mentions = target_first + rest

    return mentions


# ── Sector-Aware Source Prioritization ─────────────────────────────────

_FINANCE_DOMAINS = {
    "sec.gov": 1.0,
    "federalreserve.gov": 0.95,
    "reuters.com": 0.9,
    "bloomberg.com": 0.9,
    "ft.com": 0.85,
    "wsj.com": 0.85,
    "seekingalpha.com": 0.7,
    "yahoo.com/finance": 0.65,
    "finance.yahoo.com": 0.65,
    "marketwatch.com": 0.7,
    "investing.com": 0.6,
    "cnbc.com": 0.65,
    "morningstar.com": 0.75,
}


def prioritize_sources(
    sources: list[dict[str, Any]],
    target_ticker: str = "",
) -> list[dict[str, Any]]:
    """Re-rank sources by finance-domain authority and ticker relevance.

    Adds a ``priority_score`` field to each source dict and returns them
    sorted by descending priority.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for src in sources:
        domain = src.get("domain", "").lower()
        title = src.get("title", "").lower()

        domain_score = 0.3
        for fd, fs in _FINANCE_DOMAINS.items():
            if fd in domain:
                domain_score = fs
                break

        ticker_bonus = 0.0
        if target_ticker:
            upper = target_ticker.upper()
            if upper.lower() in title or upper in src.get("title", ""):
                ticker_bonus = 0.15

        priority = domain_score * 0.7 + ticker_bonus + 0.15
        src["priority_score"] = round(min(1.0, priority), 3)
        scored.append((src["priority_score"], src))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored]
