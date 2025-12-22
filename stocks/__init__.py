"""
Stock Advisor - Stock Expert Package

Exports:
- STOCK_EXPERTS: Dict of expert configurations
- STOCK_PRESETS: Dict of preset expert combinations
- EXPERT_ICONS: Dict of expert emoji badges
- get_stock_prompts: Generate prompts for experts
- call_stock_expert: Call an expert (sync)
- call_stock_expert_stream: Call an expert (streaming)
- detect_best_stock_expert: Route questions to best expert
"""

from .stock_personas import (
    STOCK_EXPERTS,
    STOCK_PRESETS,
    STOCK_CATEGORIES,
    EXPERT_ICONS,
    get_stock_prompts,
    get_stock_base_context,
    get_default_stock_experts,
    get_experts_by_category,
    get_all_expert_names,
    detect_best_stock_expert,
    call_stock_expert,
    call_stock_expert_stream,
)

__all__ = [
    "STOCK_EXPERTS",
    "STOCK_PRESETS",
    "STOCK_CATEGORIES",
    "EXPERT_ICONS",
    "get_stock_prompts",
    "get_stock_base_context",
    "get_default_stock_experts",
    "get_experts_by_category",
    "get_all_expert_names",
    "detect_best_stock_expert",
    "call_stock_expert",
    "call_stock_expert_stream",
]
