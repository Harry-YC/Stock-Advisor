"""
Travel Planner - Chainlit App

A conversational travel planning assistant with AI travel experts.
Uses ChatSettings for trip configuration and streaming responses.
SQLite persistence for conversation history.

Run with: chainlit run app.py
"""

import asyncio
import chainlit as cl
import chainlit.data as cl_data
from chainlit.input_widget import TextInput, Select, Slider
from datetime import date, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging
import os

import json
import re

from config import settings
from services.travel_data_service import TravelDataService
from travel.travel_personas import (
    TRAVEL_EXPERTS,
    TRAVEL_PRESETS,
    EXPERT_ICONS,
    get_default_travel_experts,
    call_travel_expert,
    call_travel_expert_stream
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM-Guided Intake Conversation
# =============================================================================

async def extract_trip_info(user_message: str, existing_info: dict) -> dict:
    """Use LLM to extract trip details from natural language."""
    from services.llm_router import get_llm_router
    router = get_llm_router()

    prompt = f"""Extract trip details from this user message. Return ONLY valid JSON, no markdown.
The user may write in English, Traditional Chinese (繁體中文), or a mix of both.

Current known info: {json.dumps(existing_info)}

User message: "{user_message}"

Return JSON with these fields (use null if not mentioned):
{{
  "destination": "city/country or null",
  "origin": "departure city/country or null",
  "departure_date": "departure/start date or null (convert Chinese dates like 一月六號 to Jan 6)",
  "return_date": "return/end date or null",
  "duration_days": number or null,
  "travelers": "description or null",
  "budget": number or null (convert Chinese numerals to digits: 一萬二 = 12000, 五萬 = 50000),
  "budget_currency": "3-letter ISO code or null. Map these terms:
    台幣/新台幣=TWD, 美金/美元=USD, 日幣/日圓/日元=JPY, 歐元=EUR, 英鎊=GBP,
    人民幣/人民币=CNY, 韓幣/韓元/韩币=KRW, 港幣/港元=HKD, 新幣/新加坡幣=SGD,
    澳幣/澳元=AUD, 加幣/加元=CAD, 瑞士法郎/瑞郎=CHF, 泰銖/泰铢=THB,
    馬幣/令吉=MYR, 披索/菲律賓披索=PHP, 印尼盾=IDR, 越南盾=VND,
    印度盧比/盧比=INR, 紐幣/紐元=NZD, 墨西哥披索=MXN, 巴西雷亞爾=BRL,
    南非蘭特=ZAR, 瑞典克朗=SEK, 挪威克朗=NOK, 丹麥克朗=DKK",
  "travel_style": "foodie/adventure/relaxation/cultural/budget/luxury or null",
  "special_interests": "specific interests mentioned or null"
}}

IMPORTANT: Pay attention to words like "returning on", "back on", "until", "回程", "返回" - these indicate the RETURN date, not departure.
If user gives both a duration and a return date, use return date and calculate departure from duration.

English Examples:
- "Barcelona next month with my wife" -> {{"destination": "Barcelona, Spain", "origin": null, "departure_date": "next month", "return_date": null, "travelers": "2 adults (couple)", "budget": null, "duration_days": null}}
- "Tokyo from San Francisco" -> {{"destination": "Tokyo, Japan", "origin": "San Francisco", "departure_date": null, "return_date": null, "travelers": null, "budget": null, "duration_days": null}}
- "3 day trip returning on 1-9-2026" -> {{"destination": null, "origin": null, "departure_date": null, "return_date": "1-9-2026", "travelers": null, "budget": null, "duration_days": 3}}
- "trip from Jan 6 to Jan 9" -> {{"destination": null, "origin": null, "departure_date": "Jan 6", "return_date": "Jan 9", "travelers": null, "budget": null, "duration_days": null}}
- "trip to Paris from Taipei budget 5000" -> {{"destination": "Paris, France", "origin": "Taipei", "departure_date": null, "return_date": null, "travelers": null, "budget": 5000, "duration_days": null}}
- "a week in mid-January" -> {{"destination": null, "origin": null, "departure_date": "mid-January", "return_date": null, "duration_days": 7, "travelers": null, "budget": null}}
- "two nights" or "for 2 nights" -> {{"duration_days": 3, ...}} (nights + 1 = total days)
- "three nights" -> {{"duration_days": 4, ...}} (nights + 1 = total days)
- "myself" or "just me" or "solo" -> {{"travelers": "1 adult (solo)"}}
- "me and my brother" or "with my sister" -> {{"travelers": "2 adults (siblings)"}}
- "foodie trip" -> extract "foodie" as a travel style hint, not travelers

Traditional Chinese Examples (繁體中文):
- "我想去東京" -> {{"destination": "Tokyo, Japan", "origin": null, ...}}
- "從台北出發去巴黎" -> {{"destination": "Paris, France", "origin": "Taipei", ...}}
- "2026年1月6日出發" or "一月六號出發" -> {{"departure_date": "Jan 6, 2026", ...}}
- "一月六號到一月九號" -> {{"departure_date": "Jan 6", "return_date": "Jan 9", ...}}
- "三天兩夜" -> {{"duration_days": 3, ...}}
- "五天四夜" -> {{"duration_days": 5, ...}}
- "一週" or "一個禮拜" -> {{"duration_days": 7, ...}}
- "我自己" or "一個人" or "獨自旅行" -> {{"travelers": "1 adult (solo)"}}
- "跟老婆" or "和太太" or "夫妻倆" -> {{"travelers": "2 adults (couple)"}}
- "全家" or "帶小孩" -> {{"travelers": "2 adults + children (family)"}}
- "預算五萬" or "預算50000" -> {{"budget": 50000, ...}}
- "預算一萬二台幣" -> {{"budget": 12000, "budget_currency": "TWD", ...}}
- "一萬二新台幣" -> {{"budget": 12000, "budget_currency": "TWD", ...}}
- "三千美金" or "3000美元" -> {{"budget": 3000, "budget_currency": "USD", ...}}
- "十萬日幣" or "十萬日圓" -> {{"budget": 100000, "budget_currency": "JPY", ...}}
- "兩萬五" -> {{"budget": 25000, ...}} (二萬五千 = 25000)
- "美食之旅" -> {{"travel_style": "foodie", ...}}
- "下個月" -> {{"departure_date": "next month", ...}}
- "明年一月" -> {{"departure_date": "January next year", ...}}

Return ONLY the JSON object, nothing else."""

    try:
        response_text = ""
        for chunk in router.call_expert_stream(
            prompt=prompt,
            system="You are a JSON extraction assistant. Return only valid JSON, no explanation."
        ):
            if chunk.get("type") == "chunk":
                response_text += chunk.get("content", "")

        # Parse JSON from response
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = json.loads(response_text.strip())

        # Merge with existing info (don't overwrite with nulls)
        result = existing_info.copy()
        for key, value in extracted.items():
            if value is not None:
                result[key] = value

        return result

    except Exception as e:
        logger.error(f"Trip info extraction failed: {e}")
        return existing_info


def get_next_question(trip_info: dict) -> Optional[str]:
    """Determine what essential info is still missing."""
    if not trip_info.get("destination"):
        return "Where would you like to go? 🌍"
    if not trip_info.get("dates") and not trip_info.get("duration_days"):
        return "When are you thinking of traveling? 📅"
    if not trip_info.get("travelers"):
        return "Who's joining you on this trip? 👥"
    if not trip_info.get("budget"):
        return "What's your approximate budget for this trip? 💰"
    return None  # All essentials collected!


async def handle_intake_message(message: str):
    """Handle messages during the intake conversation phase."""
    trip_info = cl.user_session.get("trip_info", {})

    # Show spinner while extracting info
    async with cl.Step(name="Trip Info Extractor", type="run") as step:
        trip_info = await extract_trip_info(message, trip_info)
        cl.user_session.set("trip_info", trip_info)

        # Show what we extracted
        extracted = []
        if trip_info.get("destination"):
            extracted.append(f"📍 {trip_info['destination']}")
        if trip_info.get("origin"):
            extracted.append(f"✈️ from {trip_info['origin']}")
        # Handle new date fields
        date_parts = []
        if trip_info.get("departure_date"):
            date_parts.append(trip_info['departure_date'])
        if trip_info.get("return_date"):
            date_parts.append(f"return {trip_info['return_date']}")
        if trip_info.get("duration_days"):
            date_parts.append(f"{trip_info['duration_days']} days")
        if date_parts:
            extracted.append(f"📅 {', '.join(date_parts)}")
        elif trip_info.get("dates"):
            extracted.append(f"📅 {trip_info['dates']}")
        if trip_info.get("travelers"):
            extracted.append(f"👥 {trip_info['travelers']}")
        if trip_info.get("budget"):
            # Show currency if specified (25 common currencies)
            currency = trip_info.get("budget_currency", "")
            currency_symbols = {
                "USD": "$", "TWD": "NT$", "NTD": "NT$", "JPY": "¥", "EUR": "€",
                "GBP": "£", "CNY": "¥", "RMB": "¥", "KRW": "₩", "HKD": "HK$",
                "SGD": "S$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ", "THB": "฿",
                "MYR": "RM ", "PHP": "₱", "IDR": "Rp ", "VND": "₫", "INR": "₹",
                "NZD": "NZ$", "MXN": "MX$", "BRL": "R$", "ZAR": "R ",
                "SEK": "kr ", "NOK": "kr ", "DKK": "kr "
            }
            symbol = currency_symbols.get(currency, "$")
            currency_suffix = f" {currency}" if currency and currency != "USD" else ""
            extracted.append(f"💰 {symbol}{trip_info['budget']:,}{currency_suffix}")

        step.output = " | ".join(extracted) if extracted else "Processing..."

    # Check what's still needed
    next_question = get_next_question(trip_info)

    if next_question:
        # Still need more info
        await cl.Message(content=next_question).send()
    else:
        # All essentials collected - show confirmation before planning
        await show_trip_confirmation(trip_info)


async def show_trip_confirmation(trip_info: dict):
    """Show extracted trip info and ask user to confirm before planning."""
    # Convert to config to get parsed dates
    trip_config = convert_trip_info_to_config(trip_info)

    # Build confirmation message
    destination = trip_info.get("destination", "Unknown")
    origin = trip_info.get("origin", "")
    travelers = trip_config.get("travelers", "2 adults")
    budget = trip_config.get("budget", 5000)
    budget_currency = trip_config.get("budget_currency", "USD")

    # Currency symbols (25 common currencies)
    currency_symbols = {
        "USD": "$", "TWD": "NT$", "NTD": "NT$", "JPY": "¥", "EUR": "€",
        "GBP": "£", "CNY": "¥", "RMB": "¥", "KRW": "₩", "HKD": "HK$",
        "SGD": "S$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ", "THB": "฿",
        "MYR": "RM ", "PHP": "₱", "IDR": "Rp ", "VND": "₫", "INR": "₹",
        "NZD": "NZ$", "MXN": "MX$", "BRL": "R$", "ZAR": "R ",
        "SEK": "kr ", "NOK": "kr ", "DKK": "kr "
    }
    currency_symbol = currency_symbols.get(budget_currency, budget_currency + " ")

    # Parse dates for display
    try:
        departure = date.fromisoformat(trip_config.get("departure", ""))
        return_date = date.fromisoformat(trip_config.get("return_date", ""))
        duration = (return_date - departure).days + 1
        date_str = f"{departure.strftime('%b %d')} - {return_date.strftime('%b %d, %Y')} ({duration} days)"
    except:
        date_str = "Dates need confirmation"

    # Check if we have any travel style hints
    travel_style = trip_info.get("travel_style", "")

    confirmation_msg = f"""## ✅ Here's what I understood:

| Field | Value |
|-------|-------|
| **Destination** | {destination} |
| **Origin** | {origin or 'Not specified'} |
| **Dates** | {date_str} |
| **Travelers** | {travelers} |
| **Budget** | {currency_symbol}{budget:,} {budget_currency} |
| **Trip Style** | {travel_style or 'Not specified'} |

**Does this look right?**

If something's wrong, just tell me (e.g., "actually it's just me" or "departing from Tokyo").

---

**Optional - Help me give better recommendations:**
- What's your travel style? (foodie, adventure, relaxation, cultural, budget backpacker)
- Any must-see attractions or activities?
- Dietary restrictions or accessibility needs?

*Just type any details, or click a button below to continue.*"""

    actions = [
        cl.Action(name="plan_trip", label="✅ Yes, plan my trip!", value="confirm", payload={}),
        cl.Action(name="adjust_trip", label="✏️ Let me adjust", value="adjust", payload={}),
    ]

    await cl.Message(content=confirmation_msg, actions=actions).send()


@cl.action_callback("plan_trip")
async def on_plan_trip(action: cl.Action):
    """User confirmed trip details - proceed with planning."""
    trip_info = cl.user_session.get("trip_info", {})
    trip_config = convert_trip_info_to_config(trip_info)
    cl.user_session.set("trip_config", trip_config)
    cl.user_session.set("intake_mode", False)
    await handle_plan_trip(trip_config)


@cl.action_callback("adjust_trip")
async def on_adjust_trip(action: cl.Action):
    """User wants to adjust - prompt them to make corrections."""
    await cl.Message(
        content="No problem! Just tell me what needs to be changed, for example:\n"
                "- \"Actually it's just me traveling\"\n"
                "- \"I'm flying from Tokyo\"\n"
                "- \"The dates are Jan 10-15\"\n"
                "- \"Budget is $3000\""
    ).send()


def parse_flexible_date(date_str: str, default_year: int = None) -> date:
    """
    Parse dates from various formats using dateutil with fallbacks.

    Supports:
    - ISO: 2026-01-06, 2026/1/6
    - US: 1-6-2026, 1/6/2026, 01/06/2026
    - EU: 6-1-2026, 6/1/2026 (when day > 12)
    - Month names: Jan 6 2026, 6 Jan 2026, January 6, 2026, 6Jan2026
    - Compact: 20260106
    - Relative: next week, in 2 weeks, next month
    - Chinese: 2026年1月6日, 一月六號, 一月六日
    """
    import re
    from datetime import timedelta
    from dateutil import parser as dateutil_parser

    if not date_str:
        return None

    date_str = date_str.strip()
    today = date.today()
    default_year = default_year or today.year

    # Chinese number mapping
    chinese_nums = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '兩': 2, '两': 2
    }

    def chinese_to_number(chinese_str: str) -> int:
        """Convert Chinese numerals to integer (handles 1-31)."""
        if not chinese_str:
            return None
        # Try direct digit
        if chinese_str.isdigit():
            return int(chinese_str)
        # Single character
        if len(chinese_str) == 1 and chinese_str in chinese_nums:
            return chinese_nums[chinese_str]
        # Handle tens: 十, 十一, 二十, 二十一
        if '十' in chinese_str:
            parts = chinese_str.split('十')
            tens = chinese_nums.get(parts[0], 1) if parts[0] else 1
            ones = chinese_nums.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            return tens * 10 + ones
        return None

    # Chinese month mapping
    chinese_months = {
        '一月': 1, '二月': 2, '三月': 3, '四月': 4,
        '五月': 5, '六月': 6, '七月': 7, '八月': 8,
        '九月': 9, '十月': 10, '十一月': 11, '十二月': 12
    }

    # Handle Chinese date formats: 2026年1月6日 or 1月6日 or 一月六號/日
    # Pattern: year年month月day日/號
    chinese_full = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})[日號]', date_str)
    if chinese_full:
        year, month, day = int(chinese_full.group(1)), int(chinese_full.group(2)), int(chinese_full.group(3))
        return date(year, month, day)

    # Pattern without year: 1月6日 or 1月6號
    chinese_no_year = re.search(r'(\d{1,2})月(\d{1,2})[日號]', date_str)
    if chinese_no_year:
        month, day = int(chinese_no_year.group(1)), int(chinese_no_year.group(2))
        result = date(default_year, month, day)
        if result < today:
            result = date(default_year + 1, month, day)
        return result

    # Pattern with Chinese numerals: 一月六號 or 一月六日
    for month_cn, month_num in chinese_months.items():
        if month_cn in date_str:
            # Find the day part after the month
            idx = date_str.find(month_cn) + len(month_cn)
            day_part = date_str[idx:].split('日')[0].split('號')[0]
            day = chinese_to_number(day_part)
            if day:
                # Check for year
                year_match = re.search(r'(\d{4})年', date_str)
                year = int(year_match.group(1)) if year_match else default_year
                result = date(year, month_num, day)
                if result < today and not year_match:
                    result = date(year + 1, month_num, day)
                return result

    # Handle Chinese relative dates
    if '下個月' in date_str or '下个月' in date_str:
        return date(today.year + (1 if today.month == 12 else 0),
                   (today.month % 12) + 1, min(today.day, 28))
    if '明天' in date_str:
        return today + timedelta(days=1)
    if '後天' in date_str or '后天' in date_str:
        return today + timedelta(days=2)
    if '下週' in date_str or '下周' in date_str or '下禮拜' in date_str:
        return today + timedelta(days=7)

    # Handle relative dates first (English)
    date_lower = date_str.lower()
    if 'next week' in date_lower:
        return today + timedelta(days=7)
    if 'next month' in date_lower:
        return date(today.year + (1 if today.month == 12 else 0),
                   (today.month % 12) + 1, min(today.day, 28))
    if 'tomorrow' in date_lower:
        return today + timedelta(days=1)

    # "in X days/weeks/months"
    in_match = re.search(r'in\s+(\d+)\s*(day|week|month)', date_lower)
    if in_match:
        num, unit = int(in_match.group(1)), in_match.group(2)
        if unit == 'day':
            return today + timedelta(days=num)
        elif unit == 'week':
            return today + timedelta(weeks=num)
        elif unit == 'month':
            new_month = today.month + num
            new_year = today.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            return date(new_year, new_month, min(today.day, 28))

    # Try dateutil parser first (handles many formats automatically)
    try:
        # dayfirst=False for US format (M/D/Y), yearfirst for ISO
        parsed = dateutil_parser.parse(date_str, dayfirst=False, yearfirst=False, fuzzy=True)
        result = parsed.date()
        # If year is current year but date is in past, assume next year
        if result < today and result.year == today.year:
            result = date(result.year + 1, result.month, result.day)
        return result
    except (ValueError, TypeError):
        pass

    # Manual parsing for edge cases dateutil misses

    # Compact format without separators: "6Jan2026" or "Jan62026"
    compact_match = re.search(r'(\d{1,2})([a-zA-Z]{3,9})(\d{4})', date_str)
    if compact_match:
        day_str, month_str, year_str = compact_match.groups()
        try:
            parsed = dateutil_parser.parse(f"{month_str} {day_str}, {year_str}")
            return parsed.date()
        except:
            pass

    # Reverse compact: "Jan6,2026"
    compact_match2 = re.search(r'([a-zA-Z]{3,9})(\d{1,2}),?\s*(\d{4})', date_str)
    if compact_match2:
        month_str, day_str, year_str = compact_match2.groups()
        try:
            parsed = dateutil_parser.parse(f"{month_str} {day_str}, {year_str}")
            return parsed.date()
        except:
            pass

    return None


def parse_date_range(dates_str: str, duration: int = None) -> tuple:
    """
    Parse a date range string and return (departure, return_date).

    Supports:
    - "1-6-2026 to 1-9-2026"
    - "Jan 6 - Jan 9, 2026"
    - "6-9 January 2026"
    - "from Jan 6 to Jan 9"
    - Single date with duration
    """
    import re
    from datetime import timedelta

    if not dates_str:
        # Default to 30 days from now
        departure = date.today() + timedelta(days=30)
        return_date = departure + timedelta(days=duration or 7)
        return departure, return_date

    dates_str = dates_str.strip()

    # Split on range indicators
    range_patterns = [
        r'\s+to\s+',           # "X to Y"
        r'\s*-\s*(?=[a-zA-Z])', # "X - Y" where Y starts with letter (not numeric date)
        r'\s+through\s+',      # "X through Y"
        r'\s+until\s+',        # "X until Y"
        r'\s+till\s+',         # "X till Y"
    ]

    start_str = None
    end_str = None

    for pattern in range_patterns:
        parts = re.split(pattern, dates_str, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            start_str, end_str = parts[0].strip(), parts[1].strip()
            break

    # Try to find two separate date patterns in the string
    if not start_str:
        # Pattern for dates like "1-6-2026" or "1/6/2026"
        date_patterns = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}', dates_str)
        if len(date_patterns) >= 2:
            start_str, end_str = date_patterns[0], date_patterns[1]
        elif len(date_patterns) == 1:
            start_str = date_patterns[0]

    # Parse the dates
    departure = None
    return_date = None

    if start_str:
        departure = parse_flexible_date(start_str)

    if end_str:
        # If end date doesn't have year, inherit from start
        if departure and not re.search(r'\d{4}', end_str):
            end_str = f"{end_str} {departure.year}"
        return_date = parse_flexible_date(end_str)

    # Fallbacks
    if not departure:
        departure = date.today() + timedelta(days=30)

    if not return_date:
        if duration:
            return_date = departure + timedelta(days=duration)
        else:
            return_date = departure + timedelta(days=7)

    # Sanity check: return_date should be after departure
    if return_date <= departure:
        return_date = departure + timedelta(days=duration or 7)

    return departure, return_date


def convert_trip_info_to_config(trip_info: dict) -> dict:
    """Convert intake trip_info to the trip_config format used by handle_plan_trip."""
    from datetime import timedelta

    # Get date fields from extraction
    departure_str = trip_info.get("departure_date", "") or trip_info.get("dates", "")
    return_str = trip_info.get("return_date", "")
    duration = trip_info.get("duration_days")

    # Parse dates
    departure = None
    return_date = None

    if departure_str:
        departure = parse_flexible_date(departure_str)
    if return_str:
        return_date = parse_flexible_date(return_str)

    # Calculate missing date from duration
    if return_date and not departure and duration:
        # User gave return date and duration - calculate departure
        departure = return_date - timedelta(days=duration - 1)
    elif departure and not return_date and duration:
        # User gave departure and duration - calculate return
        return_date = departure + timedelta(days=duration - 1)
    elif departure and return_date:
        # Both dates given - use them directly
        pass
    elif departure and not return_date:
        # Only departure, default to 7 days
        return_date = departure + timedelta(days=duration or 7)
    elif return_date and not departure:
        # Only return date, default to 7 days back
        departure = return_date - timedelta(days=(duration or 7) - 1)
    else:
        # No dates at all - default to 30 days from now
        departure = date.today() + timedelta(days=30)
        return_date = departure + timedelta(days=duration or 7)

    # Sanity check: return should be after departure
    if return_date <= departure:
        return_date = departure + timedelta(days=duration or 7)

    # Map travelers string to select value
    travelers_str = trip_info.get("travelers", "2 adults")
    travelers_lower = travelers_str.lower() if travelers_str else ""

    # Solo traveler indicators (English + Chinese)
    solo_indicators = [
        "1", "solo", "alone", "myself", "just me", "by myself", "single", "one person", "one adult",
        "我自己", "一個人", "一个人", "獨自", "独自", "單獨", "单独"
    ]
    if any(ind in travelers_str for ind in solo_indicators):
        travelers = "1 adult"
    # Family with children indicators (English + Chinese)
    elif any(ind in travelers_str for ind in ["child", "kid", "family", "小孩", "孩子", "全家", "兒童", "儿童", "帶小孩", "带小孩"]):
        if "2" in travelers_lower or "兩" in travelers_str or "两" in travelers_str:
            travelers = "2 adults + 2 children"
        else:
            travelers = "2 adults + 1 child"
    # Group indicators (English + Chinese)
    elif any(ind in travelers_str for ind in ["group", "4", "5", "四", "五", "團", "团", "一群"]):
        travelers = "Group (4+)"
    # Couple indicators (English + Chinese)
    elif any(ind in travelers_str for ind in ["couple", "wife", "husband", "partner", "老婆", "老公", "太太", "先生", "夫妻", "伴侶", "伴侣", "兩個人", "两个人"]):
        travelers = "2 adults"
    elif "3" in travelers_lower or "three" in travelers_lower or "三" in travelers_str:
        travelers = "Group (4+)"  # Closest match for 3 people
    else:
        travelers = "2 adults"

    # Handle budget and currency
    budget = trip_info.get("budget", 5000)
    budget_currency = trip_info.get("budget_currency", "USD")

    # If no currency specified and budget > 10000, likely TWD (common for Taiwan users)
    if not trip_info.get("budget_currency") and budget and budget > 10000:
        # Heuristic: budgets over 10000 without currency specified might be TWD
        # But we'll default to USD and let the user correct if needed
        budget_currency = "USD"

    return {
        "destination": trip_info.get("destination", ""),
        "origin": trip_info.get("origin", ""),  # Extracted from "from X" in user input
        "departure": departure.isoformat(),
        "return_date": return_date.isoformat(),
        "travelers": travelers,
        "budget": budget,
        "budget_currency": budget_currency,
        "preset": "Quick Trip Planning",
        "travel_style": trip_info.get("travel_style", ""),
        "special_interests": trip_info.get("special_interests", ""),
    }


@cl.action_callback("ask_expert")
async def on_ask_expert(action: cl.Action):
    """Handle quick expert consultation button click."""
    expert_name = action.payload.get("expert") or action.value
    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data")

    destination = trip_config.get("destination", "your destination")
    context = trip_data.get("summary", "") if trip_data else ""

    # Get previous expert responses for context
    expert_responses = cl.user_session.get("expert_responses", {})
    if expert_responses:
        context += "\n\n## Previous Expert Recommendations:\n"
        for expert, response in expert_responses.items():
            context += f"\n**{expert}:** {response[:500]}...\n"

    # Stream expert response
    icon = EXPERT_ICONS.get(expert_name, "🧭")
    msg = cl.Message(content="", author=f"{icon} {expert_name}")
    expert_info = TRAVEL_EXPERTS.get(expert_name, {})
    role = expert_info.get("role", expert_name)
    await msg.stream_token(f"## {icon} {expert_name}\n*{role}*\n\n")

    full_response = ""
    try:
        for chunk in call_travel_expert_stream(
            persona_name=expert_name,
            clinical_question=f"Provide your expert advice for a trip to {destination}",
            evidence_context=context,
            model=settings.EXPERT_MODEL,
            openai_api_key=settings.GEMINI_API_KEY
        ):
            if chunk.get("type") == "chunk":
                content = chunk.get("content", "")
                full_response += content
                await msg.stream_token(content)
            elif chunk.get("type") == "error":
                await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
    except Exception as e:
        logger.error(f"Expert {expert_name} failed: {e}")
        await msg.stream_token(f"\n\n*Error getting response: {e}*")

    await msg.send()

    # Store the new expert response
    expert_responses[expert_name] = full_response
    cl.user_session.set("expert_responses", expert_responses)


@cl.action_callback("export_excel")
async def on_export_excel(action: cl.Action):
    """Export trip plan to Excel."""
    from services.excel_export_service import export_travel_plan_to_excel
    import tempfile

    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data", {})
    expert_responses = cl.user_session.get("expert_responses", {})

    destination = trip_config.get("destination", "trip")
    safe_dest = "".join(c for c in destination if c.isalnum() or c in " -_").strip()[:30]

    await cl.Message(content="Generating your Excel trip plan...").send()

    try:
        # Build question and recommendation for parser
        question = f"Trip to {destination}, {trip_config.get('travelers', '2 adults')}, budget ${trip_config.get('budget', 5000)}"
        recommendation = trip_data.get("summary", "") if trip_data else ""

        # Convert expert_responses to expected format
        expert_data = {}
        for expert, response in expert_responses.items():
            expert_data[expert] = {"content": response} if isinstance(response, str) else response

        excel_buffer = export_travel_plan_to_excel(
            question=question,
            recommendation=recommendation,
            expert_responses=expert_data,
            trip_data=trip_data
        )

        # Save to temp file
        file_path = f"/tmp/Trip_Plan_{safe_dest}.xlsx"
        with open(file_path, "wb") as f:
            f.write(excel_buffer.getvalue())

        file = cl.File(path=file_path, name=f"Trip_Plan_{safe_dest}.xlsx")
        await cl.Message(
            content=f"Here's your trip plan for **{destination}**:",
            elements=[file]
        ).send()

    except Exception as e:
        logger.error(f"Excel export failed: {e}")
        await cl.Message(content=f"Sorry, I couldn't generate the Excel file: {e}").send()


@cl.action_callback("export_word")
async def on_export_word(action: cl.Action):
    """Export trip plan to Word document."""
    from services.word_export_service import export_travel_plan_to_word

    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data", {})
    expert_responses = cl.user_session.get("expert_responses", {})

    destination = trip_config.get("destination", "trip")
    safe_dest = "".join(c for c in destination if c.isalnum() or c in " -_").strip()[:30]

    await cl.Message(content="Generating your Word trip plan...").send()

    try:
        word_buffer = export_travel_plan_to_word(
            trip_config=trip_config,
            trip_data=trip_data,
            expert_responses=expert_responses
        )

        # Save to temp file
        file_path = f"/tmp/Trip_Plan_{safe_dest}.docx"
        with open(file_path, "wb") as f:
            f.write(word_buffer.getvalue())

        file = cl.File(path=file_path, name=f"Trip_Plan_{safe_dest}.docx")
        await cl.Message(
            content=f"Here's your trip plan for **{destination}**:",
            elements=[file]
        ).send()

    except Exception as e:
        logger.error(f"Word export failed: {e}")
        await cl.Message(content=f"Sorry, I couldn't generate the Word file: {e}").send()


# =============================================================================
# SQLite Persistence Setup
# =============================================================================

# Database path - in project directory
DB_PATH = Path(__file__).parent / "data" / "travel_planner.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Connection string for SQLite with aiosqlite
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Initialize SQLAlchemy data layer for persistence
# NOTE: Disabled temporarily due to missing table errors in Chainlit 2.3.0
# try:
#     from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
#
#     cl_data._data_layer = SQLAlchemyDataLayer(
#         conninfo=DB_URL,
#         ssl_require=False,
#     )
#     logger.info(f"SQLite persistence enabled: {DB_PATH}")
# except ImportError as e:
#     logger.warning(f"SQLAlchemy data layer not available: {e}")
#     logger.warning("Install with: pip install aiosqlite sqlalchemy")
# except Exception as e:
#     logger.warning(f"Failed to initialize SQLite persistence: {e}")
#     logger.warning("Conversations will not persist across sessions")
logger.info("Running without persistence (disabled for debugging)")

# =============================================================================
# Initialize Services
# =============================================================================

travel_service = TravelDataService()


@cl.on_chat_start
async def start():
    """Initialize chat session with conversational intake and settings fallback."""

    # Set up ChatSettings for power users (fallback option)
    settings_widgets = [
        TextInput(
            id="destination",
            label="Destination",
            placeholder="Paris, France",
            initial=""
        ),
        TextInput(
            id="origin",
            label="Origin (for flights)",
            placeholder="San Francisco, CA",
            initial=""
        ),
        TextInput(
            id="departure",
            label="Departure Date",
            placeholder="YYYY-MM-DD",
            initial=(date.today() + timedelta(days=30)).isoformat()
        ),
        TextInput(
            id="return_date",
            label="Return Date",
            placeholder="YYYY-MM-DD",
            initial=(date.today() + timedelta(days=37)).isoformat()
        ),
        Select(
            id="travelers",
            label="Travelers",
            values=["1 adult", "2 adults", "2 adults + 1 child", "2 adults + 2 children", "Group (4+)"],
            initial_value="2 adults"
        ),
        Slider(
            id="budget",
            label="Budget (USD)",
            min=500,
            max=20000,
            step=500,
            initial=5000
        ),
        Select(
            id="preset",
            label="Expert Panel",
            values=list(TRAVEL_PRESETS.keys()),
            initial_value="Quick Trip Planning"
        ),
    ]

    await cl.ChatSettings(settings_widgets).send()

    # Initialize session state
    cl.user_session.set("trip_config", {})
    cl.user_session.set("trip_data", None)
    cl.user_session.set("expert_responses", {})

    # NEW: Initialize intake mode
    cl.user_session.set("intake_mode", True)
    cl.user_session.set("trip_info", {})

    # Conversational welcome
    await cl.Message(
        content="""# ✈️ Travel Planner

Hi! I'm your AI travel planning assistant with a team of expert advisors.

**Tell me about your trip** - where are you dreaming of going? 🌍

*(Or click the ⚙️ settings icon to fill in details manually)*"""
    ).send()


@cl.on_settings_update
async def on_settings_update(settings_dict: Dict):
    """Store updated trip configuration."""
    cl.user_session.set("trip_config", settings_dict)

    # Confirm update
    destination = settings_dict.get("destination", "")
    if destination:
        await cl.Message(
            content=f"Trip settings updated: **{destination}** | "
                    f"Budget: ${settings_dict.get('budget', 5000):,} | "
                    f"Dates: {settings_dict.get('departure', 'TBD')} to {settings_dict.get('return_date', 'TBD')}"
        ).send()


async def handle_plan_update(uploaded_plan: str):
    """Update an uploaded plan with current real-time data."""
    from services.llm_router import get_llm_router

    await cl.Message(content="Updating your plan with current data...").send()

    # Extract destination from the plan
    router = get_llm_router()
    extract_prompt = f"""From this travel plan, extract ONLY the main destination city/country.
Return just the destination name, nothing else.

{uploaded_plan[:2000]}"""

    destination = ""
    for chunk in router.call_expert_stream(prompt=extract_prompt, system="Extract only the destination name."):
        if chunk.get("type") == "chunk":
            destination += chunk.get("content", "")

    destination = destination.strip()

    if destination:
        # Create trip config from extracted info
        from datetime import timedelta
        trip_config = {
            "destination": destination,
            "departure": (date.today() + timedelta(days=30)).isoformat(),
            "return_date": (date.today() + timedelta(days=37)).isoformat(),
            "travelers": "2 adults",
            "budget": 5000,
            "preset": "Quick Trip Planning"
        }
        cl.user_session.set("trip_config", trip_config)
        cl.user_session.set("intake_mode", False)

        await cl.Message(content=f"Found destination: **{destination}**. Fetching current data...").send()
        await handle_plan_trip(trip_config)
    else:
        await cl.Message(content="I couldn't identify the destination. Please tell me where you're planning to go.").send()


async def handle_plan_improvement(uploaded_plan: str):
    """Suggest improvements to an uploaded travel plan."""
    from services.llm_router import get_llm_router

    await cl.Message(content="Analyzing your plan for improvements...").send()

    router = get_llm_router()

    improvement_prompt = f"""Review this travel plan and suggest specific improvements:

{uploaded_plan[:6000]}

Provide actionable suggestions in these areas:
1. **Itinerary Optimization** - Better order, timing, or pacing
2. **Hidden Gems** - Lesser-known alternatives to tourist traps
3. **Cost Savings** - Ways to reduce expenses
4. **Logistics** - Transportation, timing, or booking tips
5. **Missing Elements** - What should be added

Be specific and practical. Reference actual places and times from their plan."""

    async with cl.Step(name="Improvement Advisor", type="tool") as step:
        improvements = ""
        for chunk in router.call_expert_stream(
            prompt=improvement_prompt,
            system="You are an experienced travel advisor. Suggest practical improvements to travel plans."
        ):
            if chunk.get("type") == "chunk":
                improvements += chunk.get("content", "")
        step.output = "Analysis complete"

    await cl.Message(content=f"## Suggested Improvements\n\n{improvements}").send()


async def handle_file_upload(message: cl.Message):
    """Handle uploaded files - extract content and offer to revise travel plans."""
    from core.utils import extract_text_from_file
    from services.llm_router import get_llm_router

    supported_types = {
        'application/pdf': 'PDF',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Excel',
        'text/plain': 'Text'
    }

    for element in message.elements:
        # Check if file type is supported
        file_type = None
        for mime, name in supported_types.items():
            if element.mime and mime in element.mime:
                file_type = name
                break

        # Also check by extension
        if not file_type and element.path:
            ext = element.path.lower().split('.')[-1]
            ext_map = {'pdf': 'PDF', 'docx': 'Word', 'xlsx': 'Excel', 'txt': 'Text'}
            file_type = ext_map.get(ext)

        if not file_type:
            await cl.Message(
                content=f"Sorry, I can't read **{element.name}**. Supported formats: PDF, Word (.docx), Excel (.xlsx), Text (.txt)"
            ).send()
            continue

        # Show processing message
        await cl.Message(content=f"Reading your {file_type} file: **{element.name}**...").send()

        # Extract text content
        async with cl.Step(name="Document Reader", type="tool") as step:
            content = extract_text_from_file(element.path, element.mime)
            step.output = f"Extracted {len(content)} characters"

        if content.startswith("Error") or content.startswith("Unsupported"):
            await cl.Message(content=f"Sorry, I couldn't read the file: {content}").send()
            continue

        # Store the uploaded content for context
        cl.user_session.set("uploaded_plan", content)
        cl.user_session.set("uploaded_filename", element.name)

        # Analyze the content with AI
        async with cl.Step(name="Plan Analyzer", type="tool") as step:
            router = get_llm_router()

            analysis_prompt = f"""Analyze this uploaded travel document and extract key information:

{content[:8000]}  # Limit to avoid token issues

Identify and summarize:
1. **Destination(s)**: Where is the trip to?
2. **Dates**: When is the trip?
3. **Itinerary**: Key activities/places planned
4. **Budget**: Any budget info mentioned
5. **Accommodations**: Hotels/stays mentioned
6. **What's missing or could be improved?**

Be concise - just extract the key points."""

            analysis = ""
            for chunk in router.call_expert_stream(
                prompt=analysis_prompt,
                system="You are a travel planning assistant. Analyze uploaded travel documents and extract key information concisely."
            ):
                if chunk.get("type") == "chunk":
                    analysis += chunk.get("content", "")

            step.output = "Analysis complete"

        # Show analysis and offer options
        response = f"""## I've read your travel plan: **{element.name}**

{analysis}

---

**What would you like me to do?**
- "Update this plan" - I'll refresh with current data (flights, hotels, weather)
- "Improve this plan" - I'll suggest enhancements
- "Just use the destination" - Start fresh planning for this destination
- Or ask me any specific questions about the plan!"""

        await cl.Message(content=response).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages."""

    trip_config = cl.user_session.get("trip_config", {})
    trip_data = cl.user_session.get("trip_data")
    intake_mode = cl.user_session.get("intake_mode", False)

    # Handle file uploads
    if message.elements:
        await handle_file_upload(message)
        return

    user_input = message.content.lower().strip()

    # Check for uploaded plan revision requests
    uploaded_plan = cl.user_session.get("uploaded_plan")
    if uploaded_plan:
        if "update" in user_input and "plan" in user_input:
            await handle_plan_update(uploaded_plan)
            return
        elif "improve" in user_input and "plan" in user_input:
            await handle_plan_improvement(uploaded_plan)
            return

    # Check for "plan my trip" command - this bypasses intake mode (power user)
    if "plan" in user_input and ("trip" in user_input or "my" in user_input):
        # If in intake mode with collected info, use that; otherwise use form settings
        if intake_mode:
            trip_info = cl.user_session.get("trip_info", {})
            if trip_info.get("destination"):
                # Convert trip_info to trip_config format
                trip_config = convert_trip_info_to_config(trip_info)
                cl.user_session.set("trip_config", trip_config)
        cl.user_session.set("intake_mode", False)
        await handle_plan_trip(trip_config)
        return

    # If in intake mode AND no trip has been planned yet, handle conversationally
    # Once trip_data exists, treat all messages as follow-ups to the planned trip
    if intake_mode and not trip_data:
        await handle_intake_message(message.content)
        return

    # Check for on-demand car rental request (only after trip is planned)
    if "car" in user_input and ("rental" in user_input or "rent" in user_input):
        await handle_car_rental_request(trip_config, trip_data)
        return

    # Check for specific expert request
    if user_input.startswith("ask "):
        await handle_expert_question(message.content, trip_config, trip_data)
        return

    # General follow-up question
    await handle_followup(message.content, trip_config, trip_data)


# =============================================================================
# Parallel Expert Execution
# =============================================================================

# Semaphore to limit concurrent API calls (prevents rate limiting)
_expert_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests
_REQUEST_DELAY = 1.5  # Seconds between requests (helps with preview model stability)


async def call_expert_async(expert_name: str, destination: str, expert_context: str) -> tuple:
    """
    Run single expert LLM call asynchronously with retry logic.

    Returns (expert_name, response_content) tuple.
    Uses semaphore to limit concurrency and retries on transient failures.
    """
    max_retries = 3
    base_delay = 2  # seconds

    async with _expert_semaphore:  # Limit concurrent requests
        # Small delay between requests (helps with preview model stability)
        await asyncio.sleep(_REQUEST_DELAY)

        for attempt in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: call_travel_expert(
                        persona_name=expert_name,
                        clinical_question=f"Plan a trip to {destination}",
                        evidence_context=expert_context,
                        openai_api_key=settings.GEMINI_API_KEY
                    )
                )
                return (expert_name, response.get('content', ''))

            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = any(x in error_msg for x in [
                    'rate limit', 'timeout', 'connection', 'server',
                    '429', '503', '502', '504', 'overloaded'
                ])

                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Expert {expert_name} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Expert {expert_name} failed after {attempt + 1} attempts: {e}")
                    return (expert_name, f"*Error: Could not reach server. Please try again.*")

        # Safety fallback - should never reach here but ensures explicit return
        return (expert_name, "*Error: Unexpected failure. Please try again.*")


async def handle_plan_trip(trip_config: Dict):
    """Execute full trip planning with expert panel."""

    destination = trip_config.get("destination", "").strip()
    if not destination:
        await cl.Message(
            content="Please set your **destination** in the settings panel first (click the gear icon)."
        ).send()
        return

    # Parse dates
    try:
        departure = date.fromisoformat(trip_config.get("departure", ""))
        return_date = date.fromisoformat(trip_config.get("return_date", ""))
    except (ValueError, TypeError):
        await cl.Message(
            content="Please set valid **departure and return dates** in the settings (format: YYYY-MM-DD)."
        ).send()
        return

    # Get expert preset
    preset_name = trip_config.get("preset", "Quick Trip Planning")
    preset = TRAVEL_PRESETS.get(preset_name, TRAVEL_PRESETS["Quick Trip Planning"])
    selected_experts = preset["experts"]

    # Build planning message with optional details
    travel_style = trip_config.get("travel_style", "")
    special_interests = trip_config.get("special_interests", "")
    origin = trip_config.get("origin", "")

    planning_lines = [
        f"## Planning your trip to {destination}\n",
        f"**Dates:** {departure.strftime('%b %d')} - {return_date.strftime('%b %d, %Y')}",
        f"**Budget:** ${trip_config.get('budget', 5000):,}",
    ]
    if origin:
        planning_lines.append(f"**Flying from:** {origin}")
    if travel_style:
        planning_lines.append(f"**Style:** {travel_style}")
    if special_interests:
        planning_lines.append(f"**Interests:** {special_interests}")
    planning_lines.append(f"**Experts:** {', '.join(selected_experts)}")
    planning_lines.append("\nFetching real-time data...")

    await cl.Message(content="\n".join(planning_lines)).send()

    # Fetch travel data
    async with cl.Step(name="Data Fetcher", type="tool") as step:
        try:
            trip_data = travel_service.fetch_travel_data(
                destination=destination,
                origin=trip_config.get("origin") or None,
                departure_date=departure,
                return_date=return_date,
                travelers=trip_config.get("travelers", "2 adults"),
                budget=int(trip_config.get("budget", 5000))
            )
            cl.user_session.set("trip_data", trip_data)

            step.output = f"Fetched: Flights, Hotels, Dining, Weather"
        except Exception as e:
            logger.error(f"Travel data fetch failed: {e}")
            trip_data = {"summary": f"Trip to {destination}"}
            step.output = f"Some data unavailable: {e}"

    # Show data summary if available
    if trip_data.get("summary"):
        await cl.Message(
            content=f"## Real-Time Data\n\n{trip_data['summary'][:2000]}"
        ).send()

    # Fetch Google Search context for experts that need real-time data
    search_contexts = {}

    # Safety Expert always gets search context (advisories, visa, health)
    if "Safety Expert" in selected_experts:
        async with cl.Step(name="Advisory Lookup", type="tool") as step:
            try:
                advisories = travel_service.fetch_safety_advisories(destination)
                if advisories:
                    search_contexts["Safety Expert"] = f"\n\n## Current Travel Advisories (Google Search)\n{advisories}"
                    step.output = "Found current travel advisories"
                else:
                    step.output = "No advisories available"
            except Exception as e:
                logger.warning(f"Safety advisory fetch failed: {e}")
                step.output = f"Advisory fetch failed: {e}"

    # For near-term trips (< 30 days), fetch current events for all experts
    days_until_departure = (departure - date.today()).days
    if days_until_departure <= 30:
        async with cl.Step(name="Event Lookup", type="tool") as step:
            try:
                events = travel_service.fetch_current_events(destination, departure)
                if events:
                    # Add events context to all experts for near-term trips
                    for expert_name in selected_experts:
                        if expert_name not in search_contexts:
                            search_contexts[expert_name] = ""
                        search_contexts[expert_name] += f"\n\n## Current Events & News (Google Search)\n{events}"
                    step.output = f"Found events/news for {destination}"
                else:
                    step.output = "No major events found"
            except Exception as e:
                logger.warning(f"Current events fetch failed: {e}")
                step.output = f"Events fetch failed: {e}"

    # Build user preferences context
    user_prefs = []
    if origin:
        user_prefs.append(f"- Flying from: {origin}")
    user_prefs.append(f"- Travelers: {trip_config.get('travelers', '2 adults')}")
    user_prefs.append(f"- Budget: ${trip_config.get('budget', 5000):,}")
    if travel_style:
        user_prefs.append(f"- Travel style: {travel_style}")
    if special_interests:
        user_prefs.append(f"- Special interests: {special_interests}")
    user_prefs_context = "\n\n## USER PREFERENCES\n" + "\n".join(user_prefs) if user_prefs else ""

    # Run experts in parallel with progress updates
    expert_responses = {}
    completed_experts = []

    # Show progress message
    expert_icons_list = [f"{EXPERT_ICONS.get(e, '🧭')} {e.split()[0]}" for e in selected_experts]
    progress_msg = await cl.Message(
        content=f"🔄 **Consulting {len(selected_experts)} experts in parallel...**\n\n"
                f"Waiting: {', '.join(expert_icons_list)}"
    ).send()

    # Helper to run expert and update progress
    async def run_expert_with_progress(expert_name: str, context: str) -> tuple:
        result = await call_expert_async(expert_name, destination, context)
        completed_experts.append(expert_name)

        # Update progress message
        done = [f"✅ {EXPERT_ICONS.get(e, '')} {e.split()[0]}" for e in completed_experts]
        pending = [f"⏳ {EXPERT_ICONS.get(e, '')} {e.split()[0]}"
                   for e in selected_experts if e not in completed_experts]
        status = " | ".join(done)
        if pending:
            status += "\n\nWaiting: " + ", ".join(pending)
        await progress_msg.update(
            content=f"🔄 **Consulting experts...**\n\n{status}"
        )
        return result

    # Build tasks for all experts
    tasks = []
    for expert_name in selected_experts:
        expert_context = user_prefs_context + "\n\n" + trip_data.get("summary", "")
        if expert_name in search_contexts:
            expert_context += search_contexts[expert_name]
        tasks.append(run_expert_with_progress(expert_name, expert_context))

    # Run all experts in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Update progress to show all done
    await progress_msg.update(content="✅ **All experts ready!** Displaying results...")

    # Display results in order
    for expert_name, response in results:
        if isinstance(response, Exception):
            logger.error(f"Expert {expert_name} failed: {response}")
            response = f"*Error: {response}*"

        icon = EXPERT_ICONS.get(expert_name, "🧭")
        expert_info = TRAVEL_EXPERTS.get(expert_name, {})
        role = expert_info.get("role", expert_name)

        content = f"## {icon} {expert_name}\n*{role}*\n\n{response}"
        await cl.Message(content=content, author=f"{icon} {expert_name}").send()
        expert_responses[expert_name] = response

    # Store responses
    cl.user_session.set("expert_responses", expert_responses)

    # Summary message with export buttons
    export_actions = [
        cl.Action(name="export_excel", label="📊 Export to Excel", value="excel", payload={"format": "excel"}),
        cl.Action(name="export_word", label="📄 Export to Word", value="word", payload={"format": "word"}),
    ]
    await cl.Message(
        content="---\n\n**Trip planning complete!** Ask me any follow-up questions about your trip, "
                "or ask a specific expert for more details (e.g., \"Ask Food & Dining Expert about restaurants\").\n\n"
                "*Need a rental car? Just say \"show car rentals\"!*",
        actions=export_actions
    ).send()

    # Suggest other experts not in the current panel
    all_experts = list(TRAVEL_EXPERTS.keys())
    other_experts = [e for e in all_experts if e not in selected_experts]

    if other_experts:
        # Build suggestion message with expert specialties
        expert_suggestions = []
        for expert in other_experts:
            icon = EXPERT_ICONS.get(expert, "🧭")
            specialty = TRAVEL_EXPERTS[expert].get("specialty", "")
            # Take first part of specialty for brevity
            short_specialty = specialty.split(",")[0] if specialty else ""
            expert_suggestions.append(f"- **{icon} {expert}**: {short_specialty}")

        # Create action buttons for quick access to other experts
        expert_actions = [
            cl.Action(
                name="ask_expert",
                label=f"{EXPERT_ICONS.get(expert, '🧭')} {expert.split()[0]}",
                value=expert,
                payload={"expert": expert}
            )
            for expert in other_experts[:4]  # Limit to 4 buttons
        ]

        await cl.Message(
            content=f"**Would you like to hear from other experts?**\n\n"
                    + "\n".join(expert_suggestions) + "\n\n"
                    "*Click a button below or say \"Ask [Expert Name] about...\"*",
            actions=expert_actions
        ).send()


async def handle_car_rental_request(trip_config: Dict, trip_data: Optional[Dict]):
    """Fetch and display car rental options on demand."""
    destination = trip_config.get("destination", "")

    if not destination:
        await cl.Message(
            content="Please tell me your destination first so I can find car rentals for you."
        ).send()
        return

    # Parse dates
    try:
        departure = date.fromisoformat(trip_config.get("departure", ""))
        return_date = date.fromisoformat(trip_config.get("return_date", ""))
    except (ValueError, TypeError):
        departure = date.today() + timedelta(days=30)
        return_date = departure + timedelta(days=7)

    await cl.Message(content=f"Looking for car rentals in {destination}...").send()

    async with cl.Step(name="Car Finder", type="tool") as step:
        try:
            car_data = travel_service.fetch_car_rentals_on_demand(
                destination=destination,
                pickup=departure,
                dropoff=return_date
            )

            if car_data and car_data != "Car rental data unavailable. Please try again later.":
                step.output = "Found car rental options"
                await cl.Message(
                    content=f"## Car Rental Options\n\n{car_data}"
                ).send()
            else:
                step.output = "No car rentals available"
                await cl.Message(
                    content="Sorry, I couldn't find car rental options for this destination. "
                           "You may want to check directly with rental agencies."
                ).send()
        except Exception as e:
            logger.error(f"Car rental fetch failed: {e}")
            step.output = f"Error: {e}"
            await cl.Message(
                content="Sorry, I had trouble fetching car rental data. Please try again later."
            ).send()


async def handle_expert_question(user_input: str, trip_config: Dict, trip_data: Optional[Dict]):
    """Handle direct question to a specific expert."""

    # Extract expert name from input
    input_lower = user_input.lower()
    expert_name = None
    question = user_input

    for name in TRAVEL_EXPERTS.keys():
        if name.lower() in input_lower:
            expert_name = name
            # Extract the question part
            idx = input_lower.find(name.lower())
            question = user_input[idx + len(name):].strip()
            if question.startswith("about"):
                question = question[5:].strip()
            break

    if not expert_name:
        await cl.Message(
            content=f"I couldn't identify which expert you want to consult. Available experts:\n\n"
                    + "\n".join(f"- **{name}**: {info['specialty']}" for name, info in TRAVEL_EXPERTS.items())
        ).send()
        return

    destination = trip_config.get("destination", "your destination")
    context = trip_data.get("summary", "") if trip_data else ""

    # Stream expert response
    icon = EXPERT_ICONS.get(expert_name, "🧭")
    msg = cl.Message(content="", author=f"{icon} {expert_name}")
    expert_info = TRAVEL_EXPERTS.get(expert_name, {})
    await msg.stream_token(f"## {icon} {expert_name}\n*Answering: {question or 'your question'}*\n\n")

    try:
        for chunk in call_travel_expert_stream(
            persona_name=expert_name,
            clinical_question=question or f"Provide detailed advice for a trip to {destination}",
            evidence_context=context,
            model=settings.EXPERT_MODEL,
            openai_api_key=settings.GEMINI_API_KEY
        ):
            if chunk.get("type") == "chunk":
                await msg.stream_token(chunk.get("content", ""))
            elif chunk.get("type") == "error":
                await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
    except Exception as e:
        await msg.stream_token(f"\n\n*Error: {e}*")

    await msg.send()


def detect_best_expert(question: str) -> Optional[str]:
    """Detect the best expert to answer a question based on keywords."""
    question_lower = question.lower()

    # Score each expert based on keyword matches
    scores = {}
    for expert_name, expert_info in TRAVEL_EXPERTS.items():
        keywords = expert_info.get("specialty_keywords", [])
        score = sum(1 for kw in keywords if kw.lower() in question_lower)
        if score > 0:
            scores[expert_name] = score

    if scores:
        # Return expert with highest score
        return max(scores, key=scores.get)
    return None


async def handle_followup(question: str, trip_config: Dict, trip_data: Optional[Dict]):
    """Handle general follow-up questions by routing to the best expert."""

    destination = trip_config.get("destination", "")
    context = ""

    if trip_data:
        context = trip_data.get("summary", "")[:3000]

    # Get previous expert responses for context
    expert_responses = cl.user_session.get("expert_responses", {})
    if expert_responses:
        context += "\n\n## Previous Expert Recommendations:\n"
        for expert, response in expert_responses.items():
            context += f"\n**{expert}:** {response[:500]}...\n"

    # Detect best expert for this question
    best_expert = detect_best_expert(question)

    if best_expert:
        # Route to specific expert with persona
        icon = EXPERT_ICONS.get(best_expert, "🧭")
        expert_info = TRAVEL_EXPERTS.get(best_expert, {})
        role = expert_info.get("role", best_expert)

        msg = cl.Message(content="", author=f"{icon} {best_expert}")
        await msg.stream_token(f"## {icon} {best_expert}\n*{role}*\n\n")

        try:
            for chunk in call_travel_expert_stream(
                persona_name=best_expert,
                clinical_question=question,
                evidence_context=context,
                model=settings.EXPERT_MODEL,
                openai_api_key=settings.GEMINI_API_KEY
            ):
                if chunk.get("type") == "chunk":
                    await msg.stream_token(chunk.get("content", ""))
                elif chunk.get("type") == "error":
                    await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
        except Exception as e:
            await msg.stream_token(f"\n\n*Error: {e}*")

        await msg.send()
    else:
        # No clear expert match - use general assistant but suggest experts
        msg = cl.Message(content="", author="🧭 Travel Assistant")

        try:
            from services.llm_router import get_llm_router
            router = get_llm_router()

            system = """You are a helpful travel planning assistant. Answer the user's question
            based on the trip context and previous expert recommendations. Be specific and practical.
            At the end of your response, suggest which specific expert they could ask for more detailed advice.
            Available experts: Budget Advisor, Logistics Planner, Safety Expert, Weather Analyst,
            Local Culture Guide, Food & Dining Expert, Activity Curator, Accommodation Specialist."""

            prompt = f"Trip: {destination or 'Not specified'}\n\nContext:\n{context}\n\nQuestion: {question}"

            for chunk in router.call_expert_stream(prompt=prompt, system=system):
                if chunk.get("type") == "chunk":
                    await msg.stream_token(chunk.get("content", ""))
                elif chunk.get("type") == "error":
                    await msg.stream_token(f"\n\n*Error: {chunk.get('content')}*")
        except Exception as e:
            await msg.stream_token(f"I apologize, I encountered an error: {e}\n\n"
                                   "Try asking a specific expert, e.g., \"Ask Budget Advisor about costs\"")

        await msg.send()


# Chainlit lifecycle hooks for persistence (optional - requires LiteralAI)
@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resume a previous chat session."""
    # Restore session state from thread metadata if available
    if thread.metadata:
        cl.user_session.set("trip_config", thread.metadata.get("trip_config", {}))
        cl.user_session.set("trip_data", thread.metadata.get("trip_data"))
        cl.user_session.set("expert_responses", thread.metadata.get("expert_responses", {}))

    await cl.Message(
        content="Welcome back! Your previous trip planning session has been restored. "
                "Continue asking questions or say \"Plan my trip\" to start fresh."
    ).send()


if __name__ == "__main__":
    # For local development, run with: chainlit run app.py
    pass
