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

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(is_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_cache(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_captured ON sentiment_cache(captured_at)")

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


# Singleton instance
_db_instance: Optional[FinancialDatabase] = None


def get_database(db_path: Optional[str] = None) -> FinancialDatabase:
    """Get or create the database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FinancialDatabase(db_path)
    return _db_instance
