"""
Alert Service

Background monitoring and notification for price alerts.
"""

import logging
import threading
from datetime import datetime, time
from typing import List, Optional, Dict, Any

from mcp_server.database import FinancialDatabase, get_database, PriceAlert
from services.email_service import EmailService, get_email_service

logger = logging.getLogger(__name__)


class AlertMonitor:
    """
    Background service for monitoring price alerts.

    Features:
    - Periodic price checking during market hours
    - Email notifications when alerts trigger
    - Thread-safe operation
    """

    # Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        db: Optional[FinancialDatabase] = None,
        email_service: Optional[EmailService] = None,
        check_interval_seconds: int = 300
    ):
        self.db = db or get_database()
        self.email = email_service or get_email_service()
        self.check_interval = check_interval_seconds
        self._scheduler = None
        self._running = False
        self._finnhub = None

    def _get_finnhub(self):
        """Lazy load Finnhub client."""
        if self._finnhub is None:
            try:
                from integrations.finnhub import FinnhubClient
                self._finnhub = FinnhubClient()
            except ImportError:
                logger.error("Finnhub client not available")
        return self._finnhub

    def start(self):
        """
        Start the background alert monitor.

        Uses APScheduler for periodic checks during market hours.
        """
        if self._running:
            logger.warning("Alert monitor already running")
            return

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("APScheduler not installed. Run: pip install APScheduler")
            return

        self._scheduler = BackgroundScheduler()

        # Check alerts every 5 minutes during market hours (Mon-Fri 9:30-16:00 ET)
        self._scheduler.add_job(
            self.check_alerts,
            CronTrigger(
                day_of_week='mon-fri',
                hour='9-15',
                minute='*/5',
                timezone='America/New_York'
            ),
            id='alert_check',
            replace_existing=True,
        )

        # Also check at 16:00 (market close)
        self._scheduler.add_job(
            self.check_alerts,
            CronTrigger(
                day_of_week='mon-fri',
                hour='16',
                minute='0',
                timezone='America/New_York'
            ),
            id='alert_check_close',
            replace_existing=True,
        )

        self._scheduler.start()
        self._running = True
        logger.info("Alert monitor started")

    def stop(self):
        """Stop the background alert monitor."""
        if self._scheduler:
            self._scheduler.shutdown()
            self._scheduler = None
        self._running = False
        logger.info("Alert monitor stopped")

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def check_alerts(self) -> List[PriceAlert]:
        """
        Check all active alerts against current prices.

        Returns:
            List of triggered alerts
        """
        finnhub = self._get_finnhub()
        if not finnhub or not finnhub.is_available():
            logger.warning("Cannot check alerts: Finnhub not available")
            return []

        alerts = self.db.get_active_alerts()
        if not alerts:
            return []

        triggered = []

        # Group alerts by symbol to minimize API calls
        symbols = set(a.symbol for a in alerts)
        prices: Dict[str, float] = {}

        for symbol in symbols:
            try:
                quote = finnhub.get_quote(symbol)
                if quote:
                    prices[symbol] = quote.current_price
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")

        # Check each alert
        for alert in alerts:
            current_price = prices.get(alert.symbol)
            if current_price is None:
                continue

            if alert.check_trigger(current_price):
                # Trigger the alert
                self.db.trigger_alert(alert.id, current_price)
                triggered.append(alert)

                logger.info(
                    f"Alert triggered: {alert.symbol} {alert.condition} "
                    f"${alert.target_price:.2f} (current: ${current_price:.2f})"
                )

                # Send notification
                self._notify_alert(alert, current_price)

        return triggered

    def _notify_alert(self, alert: PriceAlert, current_price: float):
        """Send notification for triggered alert."""
        if self.email.is_available():
            self.email.send_alert_notification(
                symbol=alert.symbol,
                condition=alert.condition,
                target_price=alert.target_price,
                current_price=current_price,
                notes=alert.notes,
            )

    def check_alert_now(self, alert_id: int) -> Optional[Dict[str, Any]]:
        """
        Manually check a specific alert.

        Args:
            alert_id: Alert ID to check

        Returns:
            Dict with check result or None
        """
        finnhub = self._get_finnhub()
        if not finnhub or not finnhub.is_available():
            return None

        # Get the specific alert
        alerts = self.db.get_all_alerts(include_triggered=True)
        alert = next((a for a in alerts if a.id == alert_id), None)

        if not alert:
            return None

        try:
            quote = finnhub.get_quote(alert.symbol)
            if not quote:
                return None

            current_price = quote.current_price
            would_trigger = alert.check_trigger(current_price)

            return {
                "alert_id": alert.id,
                "symbol": alert.symbol,
                "condition": alert.condition,
                "target_price": alert.target_price,
                "current_price": current_price,
                "is_active": alert.is_active,
                "would_trigger": would_trigger,
                "distance": current_price - alert.target_price,
                "distance_pct": ((current_price - alert.target_price) / alert.target_price * 100),
            }
        except Exception as e:
            logger.error(f"Failed to check alert {alert_id}: {e}")
            return None

    def get_alerts_summary(self) -> Dict[str, Any]:
        """
        Get summary of all alerts.

        Returns:
            Dict with alert counts and status
        """
        all_alerts = self.db.get_all_alerts(include_triggered=True)
        active_alerts = [a for a in all_alerts if a.is_active]
        triggered_alerts = [a for a in all_alerts if not a.is_active]

        return {
            "total_alerts": len(all_alerts),
            "active_alerts": len(active_alerts),
            "triggered_alerts": len(triggered_alerts),
            "symbols_monitored": len(set(a.symbol for a in active_alerts)),
            "monitor_running": self._running,
            "email_available": self.email.is_available(),
        }

    def format_alerts_status(self) -> str:
        """
        Format alerts status as markdown.

        Returns:
            Markdown string
        """
        summary = self.get_alerts_summary()
        active = self.db.get_active_alerts()

        lines = [
            "## Alert Status",
            "",
            f"**Monitor:** {'Running' if summary['monitor_running'] else 'Stopped'}",
            f"**Email Notifications:** {'Enabled' if summary['email_available'] else 'Disabled'}",
            "",
            f"Active Alerts: {summary['active_alerts']}",
            f"Triggered Today: {summary['triggered_alerts']}",
            "",
        ]

        if active:
            lines.append("### Active Alerts")
            for alert in active[:10]:  # Limit display
                lines.append(f"- {alert.format_description()}")

        return "\n".join(lines)


# Singleton instance
_alert_monitor: Optional[AlertMonitor] = None


def get_alert_monitor() -> AlertMonitor:
    """Get or create alert monitor singleton."""
    global _alert_monitor
    if _alert_monitor is None:
        _alert_monitor = AlertMonitor()
    return _alert_monitor


def start_alert_monitoring():
    """Start the background alert monitor."""
    monitor = get_alert_monitor()
    monitor.start()


def stop_alert_monitoring():
    """Stop the background alert monitor."""
    global _alert_monitor
    if _alert_monitor:
        _alert_monitor.stop()
