"""
Financial MCP Server Database

SQLite persistence for:
- Price alerts with conditions
- Portfolio positions with cost basis
- Watchlist tracking
- KOL sentiment cache
- Alert history
"""

import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "outputs" / "stock_advisor.db"


@dataclass
class PriceAlert:
    """A price alert configuration."""
    id: int
    symbol: str
    condition: str  # "above" or "below"
    target_price: float
    created_at: datetime
    triggered_at: Optional[datetime] = None
    is_active: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "condition": self.condition,
            "target_price": self.target_price,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "is_active": self.is_active,
            "notes": self.notes,
        }

    def check_trigger(self, current_price: float) -> bool:
        """Check if alert should trigger."""
        if not self.is_active:
            return False
        if self.condition == "above":
            return current_price >= self.target_price
        elif self.condition == "below":
            return current_price <= self.target_price
        return False

    def format_description(self) -> str:
        """Format alert as readable string."""
        status = "Active" if self.is_active else "Triggered"
        return f"{self.symbol} {self.condition} ${self.target_price:.2f} [{status}]"


@dataclass
class PortfolioPosition:
    """A stock position in the portfolio."""
    id: int
    symbol: str
    shares: float
    cost_basis: float  # Total cost (shares * avg price)
    avg_price: float
    added_at: datetime
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "shares": self.shares,
            "cost_basis": self.cost_basis,
            "avg_price": self.avg_price,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "notes": self.notes,
        }

    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate P&L in dollars and percentage."""
        current_value = self.shares * current_price
        pnl_dollars = current_value - self.cost_basis
        pnl_percent = (pnl_dollars / self.cost_basis * 100) if self.cost_basis > 0 else 0
        return pnl_dollars, pnl_percent

    def format_summary(self, current_price: Optional[float] = None) -> str:
        """Format position summary."""
        summary = f"{self.symbol}: {self.shares:.2f} shares @ ${self.avg_price:.2f}"
        if current_price:
            pnl_dollars, pnl_percent = self.calculate_pnl(current_price)
            pnl_sign = "+" if pnl_dollars >= 0 else ""
            summary += f" | P&L: {pnl_sign}${pnl_dollars:.2f} ({pnl_sign}{pnl_percent:.1f}%)"
        return summary


@dataclass
class WatchlistItem:
    """A ticker on the watchlist."""
    id: int
    symbol: str
    added_at: datetime
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "notes": self.notes,
            "tags": self.tags,
        }


@dataclass
class SentimentEntry:
    """Cached KOL sentiment for a ticker."""
    id: int
    symbol: str
    author: str
    platform: str
    sentiment: str  # bullish, bearish, neutral, mixed
    key_claims: List[str]
    source_url: str
    captured_at: datetime
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "author": self.author,
            "platform": self.platform,
            "sentiment": self.sentiment,
            "key_claims": self.key_claims,
            "source_url": self.source_url,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "confidence": self.confidence,
        }


@dataclass
class ChatMessage:
    """A chat message in history."""
    id: int
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    expert_responses: Optional[Dict[str, str]]
    tickers: Optional[List[str]]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "expert_responses": self.expert_responses,
            "tickers": self.tickers,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class ChatSession:
    """A chat session."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "summary": self.summary,
        }


@dataclass
class Trade:
    """A recorded trade for performance tracking."""
    id: int
    symbol: str
    action: str  # 'buy' or 'sell'
    shares: float
    price: float
    fees: float
    trade_date: datetime
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action,
            "shares": self.shares,
            "price": self.price,
            "fees": self.fees,
            "total_value": self.shares * self.price,
            "trade_date": self.trade_date.isoformat() if self.trade_date else None,
            "notes": self.notes,
        }


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot for performance tracking."""
    id: int
    date: str
    total_value: float
    total_cost: float
    daily_pnl: float
    daily_pnl_pct: float
    positions_json: str

    def to_dict(self) -> Dict[str, Any]:
        import json
        return {
            "id": self.id,
            "date": self.date,
            "total_value": self.total_value,
            "total_cost": self.total_cost,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "positions": json.loads(self.positions_json) if self.positions_json else {},
        }


@dataclass
class KOLClaim:
    """A tracked claim from a KOL."""
    id: int
    author: str
    platform: str
    ticker: str
    claim_type: str  # price_target, direction, earnings_beat, etc.
    target_price: Optional[float]
    target_date: Optional[str]
    direction: str  # bullish, bearish
    thesis: str
    source_text: str
    price_at_claim: Optional[float]
    created_at: datetime
    outcome: Optional[str] = None  # hit, miss, pending, expired
    outcome_checked_at: Optional[datetime] = None
    outcome_price: Optional[float] = None
    return_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author": self.author,
            "platform": self.platform,
            "ticker": self.ticker,
            "claim_type": self.claim_type,
            "target_price": self.target_price,
            "target_date": self.target_date,
            "direction": self.direction,
            "thesis": self.thesis,
            "source_text": self.source_text,
            "price_at_claim": self.price_at_claim,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "outcome": self.outcome,
            "outcome_checked_at": self.outcome_checked_at.isoformat() if self.outcome_checked_at else None,
            "outcome_price": self.outcome_price,
            "return_pct": self.return_pct,
        }

    def format_summary(self) -> str:
        """Format claim as readable string."""
        status = self.outcome or "pending"
        target = f"${self.target_price:.2f}" if self.target_price else self.direction
        return f"@{self.author}: {self.ticker} → {target} [{status}]"


@dataclass
class KOLScore:
    """Aggregated score for a KOL."""
    author: str
    total_claims: int
    hits: int
    misses: int
    pending: int
    expired: int
    accuracy_pct: float
    avg_return: float
    best_call: Optional[str]
    worst_call: Optional[str]
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": self.author,
            "total_claims": self.total_claims,
            "hits": self.hits,
            "misses": self.misses,
            "pending": self.pending,
            "expired": self.expired,
            "accuracy_pct": self.accuracy_pct,
            "avg_return": self.avg_return,
            "best_call": self.best_call,
            "worst_call": self.worst_call,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    def format_scorecard(self) -> str:
        """Format as markdown scorecard."""
        total_resolved = self.hits + self.misses
        accuracy_str = f"{self.accuracy_pct:.0f}%" if total_resolved > 0 else "N/A"
        return f"""📊 @{self.author} Track Record
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy: {accuracy_str} ({self.hits}/{total_resolved} calls)
Avg Return: {self.avg_return:+.1f}%
Pending: {self.pending} | Expired: {self.expired}
Best: {self.best_call or 'N/A'}
Worst: {self.worst_call or 'N/A'}"""


class FinancialDatabase:
    """
    SQLite database for financial MCP server.

    Manages alerts, portfolio, watchlist, and sentiment cache.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DEFAULT_DB_PATH)
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Price alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    condition TEXT NOT NULL CHECK(condition IN ('above', 'below')),
                    target_price REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    triggered_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    notes TEXT DEFAULT ''
                )
            """)

            # Portfolio positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    shares REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    added_at TEXT NOT NULL,
                    notes TEXT DEFAULT ''
                )
            """)

            # Watchlist table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    added_at TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    tags TEXT DEFAULT ''
                )
            """)

            # Sentiment cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    author TEXT NOT NULL,
                    platform TEXT DEFAULT '',
                    sentiment TEXT NOT NULL,
                    key_claims TEXT DEFAULT '',
                    source_url TEXT DEFAULT '',
                    captured_at TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0
                )
            """)

            # Alert history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    triggered_price REAL NOT NULL,
                    triggered_at TEXT NOT NULL,
                    FOREIGN KEY (alert_id) REFERENCES alerts(id)
                )
            """)

            # Chat sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT DEFAULT ''
                )
            """)

            # Chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    expert_responses TEXT,
                    tickers TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)

            # Trades table for performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('buy', 'sell')),
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    fees REAL DEFAULT 0,
                    trade_date TEXT NOT NULL,
                    notes TEXT DEFAULT ''
                )
            """)

            # Daily snapshots for performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_value REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    daily_pnl_pct REAL NOT NULL,
                    positions_json TEXT
                )
            """)

            # KOL claims tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kol_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author TEXT NOT NULL,
                    platform TEXT DEFAULT '',
                    ticker TEXT NOT NULL,
                    claim_type TEXT DEFAULT 'direction',
                    target_price REAL,
                    target_date TEXT,
                    direction TEXT NOT NULL,
                    thesis TEXT DEFAULT '',
                    source_text TEXT NOT NULL,
                    price_at_claim REAL,
                    created_at TEXT NOT NULL,
                    outcome TEXT DEFAULT 'pending',
                    outcome_checked_at TEXT,
                    outcome_price REAL,
                    return_pct REAL
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(is_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_cache(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_captured ON sentiment_cache(captured_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_date ON daily_snapshots(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kol_claims_author ON kol_claims(author)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kol_claims_ticker ON kol_claims(ticker)")

            logger.info(f"Database initialized at {self.db_path}")

    # =========================================================================
    # ALERTS
    # =========================================================================

    def add_alert(
        self,
        symbol: str,
        condition: str,
        target_price: float,
        notes: str = ""
    ) -> PriceAlert:
        """
        Add a new price alert.

        Args:
            symbol: Stock ticker symbol
            condition: "above" or "below"
            target_price: Price threshold
            notes: Optional notes

        Returns:
            Created PriceAlert
        """
        symbol = symbol.upper()
        condition = condition.lower()
        if condition not in ("above", "below"):
            raise ValueError(f"Invalid condition: {condition}. Use 'above' or 'below'.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO alerts (symbol, condition, target_price, created_at, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, condition, target_price, now, notes))

            alert_id = cursor.lastrowid
            logger.info(f"Created alert {alert_id}: {symbol} {condition} ${target_price}")

            return PriceAlert(
                id=alert_id,
                symbol=symbol,
                condition=condition,
                target_price=target_price,
                created_at=datetime.fromisoformat(now),
                notes=notes,
            )

    def get_active_alerts(self, symbol: Optional[str] = None) -> List[PriceAlert]:
        """
        Get active (non-triggered) alerts.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of active PriceAlert objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if symbol:
                cursor.execute("""
                    SELECT * FROM alerts WHERE is_active = 1 AND symbol = ?
                    ORDER BY created_at DESC
                """, (symbol.upper(),))
            else:
                cursor.execute("""
                    SELECT * FROM alerts WHERE is_active = 1
                    ORDER BY symbol, created_at DESC
                """)

            return [self._row_to_alert(row) for row in cursor.fetchall()]

    def get_all_alerts(self, include_triggered: bool = True) -> List[PriceAlert]:
        """Get all alerts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if include_triggered:
                cursor.execute("SELECT * FROM alerts ORDER BY symbol, created_at DESC")
            else:
                cursor.execute("SELECT * FROM alerts WHERE is_active = 1 ORDER BY symbol, created_at DESC")

            return [self._row_to_alert(row) for row in cursor.fetchall()]

    def trigger_alert(self, alert_id: int, triggered_price: float) -> bool:
        """
        Mark an alert as triggered.

        Args:
            alert_id: Alert ID
            triggered_price: Price that triggered the alert

        Returns:
            True if alert was found and triggered
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            # Get alert details
            cursor.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cursor.fetchone()
            if not row:
                return False

            # Update alert
            cursor.execute("""
                UPDATE alerts SET is_active = 0, triggered_at = ?
                WHERE id = ?
            """, (now, alert_id))

            # Log to history
            cursor.execute("""
                INSERT INTO alert_history (alert_id, symbol, condition, target_price, triggered_price, triggered_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alert_id, row["symbol"], row["condition"], row["target_price"], triggered_price, now))

            logger.info(f"Triggered alert {alert_id}: {row['symbol']} at ${triggered_price}")
            return True

    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted alert {alert_id}")
            return deleted

    def _row_to_alert(self, row: sqlite3.Row) -> PriceAlert:
        """Convert database row to PriceAlert."""
        return PriceAlert(
            id=row["id"],
            symbol=row["symbol"],
            condition=row["condition"],
            target_price=row["target_price"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            triggered_at=datetime.fromisoformat(row["triggered_at"]) if row["triggered_at"] else None,
            is_active=bool(row["is_active"]),
            notes=row["notes"] or "",
        )

    # =========================================================================
    # PORTFOLIO
    # =========================================================================

    def add_position(
        self,
        symbol: str,
        shares: float,
        price: float,
        notes: str = ""
    ) -> PortfolioPosition:
        """
        Add or update a portfolio position.

        If position exists, averages in the new shares.

        Args:
            symbol: Stock ticker
            shares: Number of shares to add
            price: Purchase price per share
            notes: Optional notes

        Returns:
            Updated PortfolioPosition
        """
        symbol = symbol.upper()
        cost = shares * price

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check for existing position
            cursor.execute("SELECT * FROM portfolio WHERE symbol = ?", (symbol,))
            existing = cursor.fetchone()

            if existing:
                # Average in new shares
                new_shares = existing["shares"] + shares
                new_cost_basis = existing["cost_basis"] + cost
                new_avg_price = new_cost_basis / new_shares if new_shares > 0 else 0

                cursor.execute("""
                    UPDATE portfolio SET shares = ?, cost_basis = ?, avg_price = ?, notes = ?
                    WHERE symbol = ?
                """, (new_shares, new_cost_basis, new_avg_price, notes or existing["notes"], symbol))

                position_id = existing["id"]
                added_at = datetime.fromisoformat(existing["added_at"])
                logger.info(f"Updated position {symbol}: +{shares} shares @ ${price}")
            else:
                # Create new position
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO portfolio (symbol, shares, cost_basis, avg_price, added_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, shares, cost, price, now, notes))

                position_id = cursor.lastrowid
                added_at = datetime.fromisoformat(now)
                logger.info(f"Created position {symbol}: {shares} shares @ ${price}")

            # Fetch final state
            cursor.execute("SELECT * FROM portfolio WHERE id = ?", (position_id,))
            row = cursor.fetchone()
            return self._row_to_position(row)

    def reduce_position(self, symbol: str, shares: float) -> Optional[PortfolioPosition]:
        """
        Reduce (sell) shares from a position.

        Args:
            symbol: Stock ticker
            shares: Number of shares to sell

        Returns:
            Updated position, or None if position closed
        """
        symbol = symbol.upper()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM portfolio WHERE symbol = ?", (symbol,))
            existing = cursor.fetchone()

            if not existing:
                raise ValueError(f"No position found for {symbol}")

            new_shares = existing["shares"] - shares

            if new_shares <= 0:
                # Close position entirely
                cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
                logger.info(f"Closed position {symbol}")
                return None
            else:
                # Reduce proportionally
                ratio = new_shares / existing["shares"]
                new_cost_basis = existing["cost_basis"] * ratio

                cursor.execute("""
                    UPDATE portfolio SET shares = ?, cost_basis = ?
                    WHERE symbol = ?
                """, (new_shares, new_cost_basis, symbol))

                cursor.execute("SELECT * FROM portfolio WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                logger.info(f"Reduced position {symbol}: -{shares} shares")
                return self._row_to_position(row)

    def get_portfolio(self) -> List[PortfolioPosition]:
        """Get all portfolio positions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM portfolio ORDER BY symbol")
            return [self._row_to_position(row) for row in cursor.fetchall()]

    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Get a specific position."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM portfolio WHERE symbol = ?", (symbol.upper(),))
            row = cursor.fetchone()
            return self._row_to_position(row) if row else None

    def delete_position(self, symbol: str) -> bool:
        """Delete a portfolio position entirely."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol.upper(),))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted position {symbol}")
            return deleted

    def _row_to_position(self, row: sqlite3.Row) -> PortfolioPosition:
        """Convert database row to PortfolioPosition."""
        return PortfolioPosition(
            id=row["id"],
            symbol=row["symbol"],
            shares=row["shares"],
            cost_basis=row["cost_basis"],
            avg_price=row["avg_price"],
            added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
            notes=row["notes"] or "",
        )

    # =========================================================================
    # WATCHLIST
    # =========================================================================

    def add_to_watchlist(
        self,
        symbol: str,
        notes: str = "",
        tags: Optional[List[str]] = None
    ) -> WatchlistItem:
        """Add a ticker to the watchlist."""
        symbol = symbol.upper()
        tags = tags or []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if already exists
            cursor.execute("SELECT * FROM watchlist WHERE symbol = ?", (symbol,))
            if cursor.fetchone():
                raise ValueError(f"{symbol} already in watchlist")

            now = datetime.now().isoformat()
            tags_str = ",".join(tags)

            cursor.execute("""
                INSERT INTO watchlist (symbol, added_at, notes, tags)
                VALUES (?, ?, ?, ?)
            """, (symbol, now, notes, tags_str))

            item_id = cursor.lastrowid
            logger.info(f"Added {symbol} to watchlist")

            return WatchlistItem(
                id=item_id,
                symbol=symbol,
                added_at=datetime.fromisoformat(now),
                notes=notes,
                tags=tags,
            )

    def get_watchlist(self) -> List[WatchlistItem]:
        """Get all watchlist items."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watchlist ORDER BY symbol")

            items = []
            for row in cursor.fetchall():
                tags = row["tags"].split(",") if row["tags"] else []
                items.append(WatchlistItem(
                    id=row["id"],
                    symbol=row["symbol"],
                    added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
                    notes=row["notes"] or "",
                    tags=tags,
                ))
            return items

    def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove a ticker from watchlist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Removed {symbol} from watchlist")
            return deleted

    # =========================================================================
    # SENTIMENT CACHE
    # =========================================================================

    def add_sentiment(
        self,
        symbol: str,
        author: str,
        sentiment: str,
        key_claims: List[str],
        platform: str = "",
        source_url: str = "",
        confidence: float = 0.0
    ) -> SentimentEntry:
        """Add a sentiment entry from KOL analysis."""
        symbol = symbol.upper()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            claims_str = "|||".join(key_claims)  # Use ||| as separator

            cursor.execute("""
                INSERT INTO sentiment_cache
                (symbol, author, platform, sentiment, key_claims, source_url, captured_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, author, platform, sentiment, claims_str, source_url, now, confidence))

            entry_id = cursor.lastrowid
            logger.info(f"Added sentiment for {symbol} from {author}")

            return SentimentEntry(
                id=entry_id,
                symbol=symbol,
                author=author,
                platform=platform,
                sentiment=sentiment,
                key_claims=key_claims,
                source_url=source_url,
                captured_at=datetime.fromisoformat(now),
                confidence=confidence,
            )

    def get_sentiment(
        self,
        symbol: str,
        days: int = 7
    ) -> List[SentimentEntry]:
        """Get sentiment entries for a symbol from the last N days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT * FROM sentiment_cache
                WHERE symbol = ? AND captured_at >= ?
                ORDER BY captured_at DESC
            """, (symbol.upper(), cutoff))

            entries = []
            for row in cursor.fetchall():
                claims = row["key_claims"].split("|||") if row["key_claims"] else []
                entries.append(SentimentEntry(
                    id=row["id"],
                    symbol=row["symbol"],
                    author=row["author"],
                    platform=row["platform"] or "",
                    sentiment=row["sentiment"],
                    key_claims=claims,
                    source_url=row["source_url"] or "",
                    captured_at=datetime.fromisoformat(row["captured_at"]) if row["captured_at"] else None,
                    confidence=row["confidence"] or 0.0,
                ))
            return entries

    def get_sentiment_summary(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get aggregated sentiment summary for a symbol."""
        entries = self.get_sentiment(symbol, days)

        if not entries:
            return {
                "symbol": symbol.upper(),
                "total_entries": 0,
                "bullish": 0,
                "bearish": 0,
                "neutral": 0,
                "mixed": 0,
                "dominant_sentiment": "unknown",
                "key_claims": [],
            }

        counts = {"bullish": 0, "bearish": 0, "neutral": 0, "mixed": 0}
        all_claims = []

        for entry in entries:
            s = entry.sentiment.lower()
            if s in counts:
                counts[s] += 1
            all_claims.extend(entry.key_claims)

        dominant = max(counts, key=counts.get)

        return {
            "symbol": symbol.upper(),
            "total_entries": len(entries),
            "bullish": counts["bullish"],
            "bearish": counts["bearish"],
            "neutral": counts["neutral"],
            "mixed": counts["mixed"],
            "dominant_sentiment": dominant,
            "key_claims": list(set(all_claims))[:10],  # Top 10 unique claims
        }

    def cleanup_old_sentiment(self, days: int = 30):
        """Remove sentiment entries older than N days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("DELETE FROM sentiment_cache WHERE captured_at < ?", (cutoff,))
            deleted = cursor.rowcount
            if deleted:
                logger.info(f"Cleaned up {deleted} old sentiment entries")

    # =========================================================================
    # CHAT HISTORY
    # =========================================================================

    def create_session(self, session_id: str, title: str = "") -> ChatSession:
        """Create a new chat session."""
        import json
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO sessions (id, title, created_at, updated_at, summary)
                VALUES (?, ?, ?, ?, '')
            """, (session_id, title or "New Session", now, now))

            return ChatSession(
                id=session_id,
                title=title or "New Session",
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now),
                summary="",
            )

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        expert_responses: Optional[Dict[str, str]] = None,
        tickers: Optional[List[str]] = None
    ) -> ChatMessage:
        """Save a chat message to history."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            # Ensure session exists
            cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO sessions (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, content[:50] if content else "New Session", now, now))

            # Update session timestamp and title if first message
            cursor.execute("""
                UPDATE sessions SET updated_at = ?,
                    title = CASE WHEN title = 'New Session' OR title = '' THEN ? ELSE title END
                WHERE id = ?
            """, (now, content[:50] if content else "New Session", session_id))

            # Save message
            expert_json = json.dumps(expert_responses) if expert_responses else None
            tickers_json = json.dumps(tickers) if tickers else None

            cursor.execute("""
                INSERT INTO chat_history (session_id, role, content, expert_responses, tickers, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, role, content, expert_json, tickers_json, now))

            msg_id = cursor.lastrowid
            return ChatMessage(
                id=msg_id,
                session_id=session_id,
                role=role,
                content=content,
                expert_responses=expert_responses,
                tickers=tickers,
                timestamp=datetime.fromisoformat(now),
            )

    def get_session_history(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get chat history for a session."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))

            messages = []
            for row in cursor.fetchall():
                messages.append(ChatMessage(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    content=row["content"],
                    expert_responses=json.loads(row["expert_responses"]) if row["expert_responses"] else None,
                    tickers=json.loads(row["tickers"]) if row["tickers"] else None,
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                ))
            return list(reversed(messages))  # Oldest first

    def list_sessions(self, limit: int = 20) -> List[ChatSession]:
        """List recent chat sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))

            return [ChatSession(
                id=row["id"],
                title=row["title"] or "",
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                summary=row["summary"] or "",
            ) for row in cursor.fetchall()]

    def update_session_summary(self, session_id: str, summary: str) -> bool:
        """Update session summary (AI-generated)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions SET summary = ? WHERE id = ?
            """, (summary, session_id))
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    # =========================================================================
    # TRADES & PERFORMANCE
    # =========================================================================

    def record_trade(
        self,
        symbol: str,
        action: str,
        shares: float,
        price: float,
        fees: float = 0,
        trade_date: Optional[datetime] = None,
        notes: str = ""
    ) -> Trade:
        """Record a trade for performance tracking."""
        symbol = symbol.upper()
        action = action.lower()
        if action not in ("buy", "sell"):
            raise ValueError(f"Invalid action: {action}. Use 'buy' or 'sell'.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            trade_dt = (trade_date or datetime.now()).isoformat()

            cursor.execute("""
                INSERT INTO trades (symbol, action, shares, price, fees, trade_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, shares, price, fees, trade_dt, notes))

            trade_id = cursor.lastrowid
            logger.info(f"Recorded trade: {action} {shares} {symbol} @ ${price}")

            return Trade(
                id=trade_id,
                symbol=symbol,
                action=action,
                shares=shares,
                price=price,
                fees=fees,
                trade_date=datetime.fromisoformat(trade_dt),
                notes=notes,
            )

    def get_trades(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[Trade]:
        """Get trade history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM trades"
            params = []
            conditions = []

            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol.upper())

            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                conditions.append("trade_date >= ?")
                params.append(cutoff)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY trade_date DESC"
            cursor.execute(query, params)

            return [Trade(
                id=row["id"],
                symbol=row["symbol"],
                action=row["action"],
                shares=row["shares"],
                price=row["price"],
                fees=row["fees"],
                trade_date=datetime.fromisoformat(row["trade_date"]) if row["trade_date"] else None,
                notes=row["notes"] or "",
            ) for row in cursor.fetchall()]

    def save_daily_snapshot(
        self,
        total_value: float,
        total_cost: float,
        positions: Dict[str, Any]
    ) -> DailySnapshot:
        """Save daily portfolio snapshot."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")

            # Calculate daily P&L
            daily_pnl = total_value - total_cost
            daily_pnl_pct = (daily_pnl / total_cost * 100) if total_cost > 0 else 0

            positions_json = json.dumps(positions)

            cursor.execute("""
                INSERT OR REPLACE INTO daily_snapshots
                (date, total_value, total_cost, daily_pnl, daily_pnl_pct, positions_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (today, total_value, total_cost, daily_pnl, daily_pnl_pct, positions_json))

            snapshot_id = cursor.lastrowid
            logger.info(f"Saved daily snapshot: ${total_value:.2f} ({daily_pnl_pct:+.2f}%)")

            return DailySnapshot(
                id=snapshot_id,
                date=today,
                total_value=total_value,
                total_cost=total_cost,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                positions_json=positions_json,
            )

    def get_snapshots(self, days: int = 30) -> List[DailySnapshot]:
        """Get daily snapshots for the last N days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            cursor.execute("""
                SELECT * FROM daily_snapshots
                WHERE date >= ?
                ORDER BY date ASC
            """, (cutoff,))

            return [DailySnapshot(
                id=row["id"],
                date=row["date"],
                total_value=row["total_value"],
                total_cost=row["total_cost"],
                daily_pnl=row["daily_pnl"],
                daily_pnl_pct=row["daily_pnl_pct"],
                positions_json=row["positions_json"] or "{}",
            ) for row in cursor.fetchall()]

    # =========================================================================
    # KOL CLAIMS
    # =========================================================================

    def save_kol_claim(
        self,
        author: str,
        ticker: str,
        direction: str,
        source_text: str,
        platform: str = "",
        claim_type: str = "direction",
        target_price: Optional[float] = None,
        target_date: Optional[str] = None,
        thesis: str = "",
        price_at_claim: Optional[float] = None,
    ) -> KOLClaim:
        """Save a KOL claim for tracking."""
        ticker = ticker.upper()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO kol_claims
                (author, platform, ticker, claim_type, target_price, target_date,
                 direction, thesis, source_text, price_at_claim, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (author, platform, ticker, claim_type, target_price, target_date,
                  direction, thesis, source_text, price_at_claim, now))

            claim_id = cursor.lastrowid
            logger.info(f"Saved KOL claim: @{author} on {ticker} ({direction})")

            return KOLClaim(
                id=claim_id,
                author=author,
                platform=platform,
                ticker=ticker,
                claim_type=claim_type,
                target_price=target_price,
                target_date=target_date,
                direction=direction,
                thesis=thesis,
                source_text=source_text,
                price_at_claim=price_at_claim,
                created_at=datetime.fromisoformat(now),
            )

    def get_kol_claims(
        self,
        author: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 50
    ) -> List[KOLClaim]:
        """Get KOL claims with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM kol_claims"
            params = []
            conditions = []

            if author:
                conditions.append("author = ?")
                params.append(author)

            if ticker:
                conditions.append("ticker = ?")
                params.append(ticker.upper())

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            return [self._row_to_kol_claim(row) for row in cursor.fetchall()]

    def get_author_stats(self, author: str) -> Dict[str, Any]:
        """Get basic stats for a KOL author."""
        claims = self.get_kol_claims(author=author, limit=100)

        if not claims:
            return {
                "author": author,
                "total_claims": 0,
                "tickers": [],
                "directions": {"bullish": 0, "bearish": 0},
            }

        tickers = list(set(c.ticker for c in claims))
        directions = {"bullish": 0, "bearish": 0}
        for c in claims:
            if c.direction.lower() in directions:
                directions[c.direction.lower()] += 1

        return {
            "author": author,
            "total_claims": len(claims),
            "tickers": tickers,
            "directions": directions,
            "first_claim": claims[-1].created_at.isoformat() if claims else None,
            "last_claim": claims[0].created_at.isoformat() if claims else None,
        }

    def _row_to_kol_claim(self, row: sqlite3.Row) -> KOLClaim:
        """Convert database row to KOLClaim."""
        return KOLClaim(
            id=row["id"],
            author=row["author"],
            platform=row["platform"] or "",
            ticker=row["ticker"],
            claim_type=row["claim_type"] or "direction",
            target_price=row["target_price"],
            target_date=row["target_date"],
            direction=row["direction"],
            thesis=row["thesis"] or "",
            source_text=row["source_text"],
            price_at_claim=row["price_at_claim"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            outcome=row["outcome"],
            outcome_checked_at=datetime.fromisoformat(row["outcome_checked_at"]) if row["outcome_checked_at"] else None,
            outcome_price=row["outcome_price"],
            return_pct=row["return_pct"],
        )


# Singleton instance
_db_instance: Optional[FinancialDatabase] = None


def get_database(db_path: Optional[str] = None) -> FinancialDatabase:
    """Get or create the database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FinancialDatabase(db_path)
    return _db_instance
