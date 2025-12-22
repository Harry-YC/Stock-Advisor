"""
Email Service

Sends email notifications for price alerts and daily digests.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """
    Service for sending email notifications.

    Supports:
    - Price alert notifications
    - Daily portfolio digests
    - Attachments (PDF reports, etc.)
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        default_recipient: Optional[str] = None
    ):
        self.smtp_host = smtp_host or getattr(settings, 'SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = smtp_port or getattr(settings, 'SMTP_PORT', 587)
        self.smtp_user = smtp_user or getattr(settings, 'SMTP_USER', None)
        self.smtp_password = smtp_password or getattr(settings, 'SMTP_PASSWORD', None)
        self.default_recipient = default_recipient or getattr(settings, 'ALERT_EMAIL', None)

    def is_available(self) -> bool:
        """Check if email service is configured."""
        return bool(self.smtp_user and self.smtp_password)

    def send_email(
        self,
        subject: str,
        body: str,
        recipient: Optional[str] = None,
        html_body: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """
        Send an email.

        Args:
            subject: Email subject
            body: Plain text body
            recipient: Recipient email (default: configured alert email)
            html_body: Optional HTML body
            attachments: Optional list of file paths to attach

        Returns:
            True if sent successfully
        """
        if not self.is_available():
            logger.warning("Email service not configured")
            return False

        recipient = recipient or self.default_recipient
        if not recipient:
            logger.warning("No recipient specified")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = recipient

            # Add plain text
            msg.attach(MIMEText(body, 'plain'))

            # Add HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))

            # Add attachments
            if attachments:
                for filepath in attachments:
                    path = Path(filepath)
                    if path.exists():
                        with open(path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename={path.name}'
                            )
                            msg.attach(part)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_alert_notification(
        self,
        symbol: str,
        condition: str,
        target_price: float,
        current_price: float,
        notes: str = ""
    ) -> bool:
        """
        Send price alert notification.

        Args:
            symbol: Stock ticker
            condition: 'above' or 'below'
            target_price: Alert trigger price
            current_price: Current stock price
            notes: Optional notes

        Returns:
            True if sent
        """
        subject = f"Stock Alert: {symbol} {condition} ${target_price:.2f}"

        body = f"""
Stock Price Alert Triggered!

Symbol: {symbol}
Condition: Price {condition} ${target_price:.2f}
Current Price: ${current_price:.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{f'Notes: {notes}' if notes else ''}

This alert was triggered automatically by Stock Advisor.
"""

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
<h2 style="color: {'#4CAF50' if condition == 'above' else '#F44336'};">
    Stock Price Alert Triggered!
</h2>
<table style="border-collapse: collapse; margin: 20px 0;">
    <tr>
        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Symbol</strong></td>
        <td style="padding: 10px; border: 1px solid #ddd;">{symbol}</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Condition</strong></td>
        <td style="padding: 10px; border: 1px solid #ddd;">Price {condition} ${target_price:.2f}</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Current Price</strong></td>
        <td style="padding: 10px; border: 1px solid #ddd;">${current_price:.2f}</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #ddd;"><strong>Time</strong></td>
        <td style="padding: 10px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
    </tr>
</table>
{f'<p><strong>Notes:</strong> {notes}</p>' if notes else ''}
<p style="color: #666; font-size: 12px;">
    This alert was triggered automatically by Stock Advisor.
</p>
</body>
</html>
"""

        return self.send_email(subject, body, html_body=html_body)

    def send_daily_digest(
        self,
        portfolio_summary: str,
        triggered_alerts: List[Dict[str, Any]],
        market_summary: Optional[str] = None,
        report_path: Optional[str] = None
    ) -> bool:
        """
        Send daily portfolio digest.

        Args:
            portfolio_summary: Markdown summary of portfolio
            triggered_alerts: List of triggered alerts
            market_summary: Optional market overview
            report_path: Optional path to PDF report

        Returns:
            True if sent
        """
        date_str = datetime.now().strftime('%Y-%m-%d')
        subject = f"Stock Advisor Daily Digest - {date_str}"

        # Build plain text body
        body_parts = [
            f"Stock Advisor Daily Digest",
            f"Date: {date_str}",
            "",
            "=" * 50,
            "PORTFOLIO SUMMARY",
            "=" * 50,
            portfolio_summary,
            "",
        ]

        if triggered_alerts:
            body_parts.extend([
                "=" * 50,
                "TRIGGERED ALERTS",
                "=" * 50,
            ])
            for alert in triggered_alerts:
                body_parts.append(
                    f"- {alert['symbol']}: {alert['condition']} ${alert['target_price']:.2f} "
                    f"(triggered at ${alert.get('triggered_price', 0):.2f})"
                )
            body_parts.append("")

        if market_summary:
            body_parts.extend([
                "=" * 50,
                "MARKET OVERVIEW",
                "=" * 50,
                market_summary,
            ])

        body = "\n".join(body_parts)

        attachments = [report_path] if report_path else None

        return self.send_email(subject, body, attachments=attachments)

    def send_test_email(self) -> bool:
        """
        Send a test email to verify configuration.

        Returns:
            True if sent successfully
        """
        subject = "Stock Advisor - Test Email"
        body = f"""
This is a test email from Stock Advisor.

If you received this email, your email notifications are configured correctly.

Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_email(subject, body)


# Convenience function
def get_email_service() -> EmailService:
    """Get email service instance."""
    return EmailService()
