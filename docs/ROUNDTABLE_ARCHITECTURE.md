# Deliberative Roundtable Architecture

A sophisticated multi-round expert deliberation system for generating high-quality outputs through structured disagreement, cross-pollination, and stress-testing.

## Overview

This architecture implements a **deliberative AI panel** where multiple expert personas engage in structured rounds of critique, ideation, challenge-response dialogue, and convergence voting. Unlike single-pass LLM calls, this produces outputs that have been stress-tested from multiple perspectives.

**Key Innovation**: Experts are forced to interact with each other's ideas, not just respond to the prompt independently.

## Design Pattern

### Current (Optimized for Speed)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DELIBERATIVE ROUNDTABLE (Fast Mode)              │
├─────────────────────────────────────────────────────────────────────┤
│  Round 1: CRITIQUE (Discovery)                    [Flash, Parallel] │
│    └─ 8 experts analyze problem independently                       │
│    └─ No solutions yet - just identify failure modes                │
│                                                                     │
│  Round 2: BRAINSTORM (Divergent)                  [Flash, Parallel] │
│    └─ 8 experts generate solutions independently                    │
│    └─ Seeded with Round 1 findings                                  │
│    └─ No convergence yet - maximize diversity                       │
│                                                                     │
│  Round 3: CROSS-POLLINATE (Interaction)           [Flash, Parallel] │
│    └─ Experts build on each other's proposals                       │
│    └─ Extract cross-pollination links                               │
│                                                                     │
│  Round 4: CONVERGE (Voting)                       [Pro, Parallel]   │
│    └─ Chair synthesizes top 3 candidates                            │
│    └─ All experts vote with reasoning                               │
│    └─ Dissenting views explicitly captured                          │
│                                                                     │
│  Round 5: STRESS TEST (Validation)                [Pro, Parallel]   │
│    └─ Patient persona simulations                                   │
│    └─ Failure mode analysis                                         │
│    └─ Consent clarity assessment                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Estimated time: ~40-60 seconds** (8 experts, 5 rounds)

### Full Mode (Disabled for Speed)

The following sub-rounds can be re-enabled for deeper deliberation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ADDITIONAL SUB-ROUNDS (Disabled)                 │
├─────────────────────────────────────────────────────────────────────┤
│  Round 1.5: META-REVIEW (Synthesis)               [Sequential]      │
│    └─ Deduplicate findings across experts                           │
│    └─ Score by novelty/severity/likelihood                          │
│    └─ Select top 5 hypotheses                                       │
│                                                                     │
│  Round 3a: CHALLENGE                              [Sequential]      │
│    └─ Experts critique each other's proposals                       │
│                                                                     │
│  Round 3b: RESPOND                                [Sequential]      │
│    └─ Originators defend/refine proposals                           │
│                                                                     │
│  Round 3c: BLIND SPOT CHECK                       [Sequential]      │
│    └─ Devil's advocate analysis                                     │
│                                                                     │
│  Round 4.5: EVOLVE (Optimization)                 [Sequential]      │
│    └─ Generate variants optimized for different objectives          │
│    └─ Trade-off analysis between competing values                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Additional time if enabled: +2-4 minutes**

## Data Models

### Input
```python
@dataclass
class RoundtableInput:
    term: str                      # The thing being evaluated
    clinical_context: str          # Context for evaluation
    experts: List[str]             # Selected expert personas
    num_rounds: int = 5            # 1=critique, 3=with ideation, 5=full
    discovery_mode: bool = True    # Let experts find problems vs seed them
    known_issues: str = ""         # Pre-seed issues (if discovery_mode=False)
```

### Core Output Models

```python
@dataclass
class LimitationMap:
    """Failure mode analysis."""
    ambiguity_sources: List[str]
    misinterpretation_pathways: List[Dict[str, str]]  # {trigger, conclusion}
    stakeholder_harms: Dict[str, List[str]]           # {stakeholder: [harms]}
    consent_risks: List[str]

@dataclass
class CandidatePhrase:
    """A candidate solution with full analysis."""
    phrase: str
    intent_category: str           # hope-forward, risk-forward, neutral, values-based
    pros: List[str]
    cons: List[str]
    when_to_use: str
    when_not_to_use: str
    expert_votes: Dict[str, str]   # {expert: reasoning}
    rank: int

@dataclass
class DissentingView:
    """Minority dissent from voting."""
    expert: str
    position: str                  # What they disagree with
    reasoning: str
    alternative: str
    potential_harm: str            # Risk of ignoring this dissent

@dataclass
class DialogueExchange:
    """Challenge-response between experts."""
    challenger: str
    target_expert: str
    target_proposal: str
    challenge: str
    response: str
    outcome: str                   # revised, defended, conceded, partial

@dataclass
class BlindSpot:
    """Shared assumption detected by devil's advocate."""
    detected_by: str
    assumption: str
    risk: str
    affected_populations: List[str]
    mitigation: str

@dataclass
class ScoredFinding:
    """Finding with novelty/severity scores."""
    description: str
    raised_by: List[str]
    novelty: int                   # 1-5
    severity: int                  # 1-5
    likelihood: int                # 1-5
    composite_score: float
```

### Full Output
```python
@dataclass
class RoundtableOutput:
    # Core outputs
    limitation_map: LimitationMap
    idea_bank: IdeaBank
    top_3_candidates: List[CandidatePhrase]

    # Disagreement surfacing
    dissenting_views: List[DissentingView]
    challenge_responses: List[DialogueExchange]
    blind_spots: List[BlindSpot]

    # Process artifacts
    round_transcripts: Dict[str, Dict[str, str]]
    cross_pollination_links: List[Dict[str, str]]

    # Metadata
    experts_consulted: List[str]
    session_duration_seconds: float
    status: str  # critique_only, brainstorm_complete, complete
```

## Expert Constraint System (Lane Enforcement)

Prevents experts from opining outside their expertise:

```python
EXPERT_CONSTRAINTS = {
    "Surgical Oncologist": {
        "must_focus": ["Technical feasibility", "Surgical outcomes"],
        "must_avoid": ["Cognitive bias analysis", "Plain language rewrites"]
    },
    "Behavioral Decision Scientist": {
        "must_focus": ["Cognitive biases", "Framing effects", "Decision architecture"],
        "must_avoid": ["Clinical protocols", "Treatment recommendations"]
    },
    # ...
}
```

This is injected into each expert's system prompt to prevent lane-crossing.

## Patient Persona Simulation

Stress-tests outputs against realistic patient archetypes:

```python
PATIENT_PERSONAS = [
    {
        "name": "Maria, 68",
        "profile": "Spanish-speaking, 4th grade education, trusts doctors implicitly",
        "test_question": "How would Maria explain this to her daughter?",
        "failure_modes": ["binary thinking", "over-trust", "translation loss"]
    },
    {
        "name": "James, 52",
        "profile": "MBA, researches everything online, skeptical of optimistic framing",
        "test_question": "What would James Google after hearing this?",
        "failure_modes": ["over-analysis", "trust erosion"]
    },
    {
        "name": "Priya, 45",
        "profile": "Family history of cancer deaths, high anxiety, catastrophizes",
        "test_question": "What's the worst-case interpretation Priya would land on?",
        "failure_modes": ["catastrophizing", "selective hearing", "anxiety spiral"]
    }
]
```

## Parallelization Strategy

```python
def _run_experts_parallel(
    self,
    experts: List[str],
    prompt_builder: callable,
    max_workers: int = 8,
    use_fast_model: bool = False
) -> Dict[str, str]:
    """Run multiple expert LLM calls in parallel."""
    model_to_use = self.fast_model if use_fast_model else self.model

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_expert, e): e for e in experts}
        for future in as_completed(futures):
            expert, response = future.result()
            results[expert] = response
    return results
```

**Current configuration:**
- `max_workers=8` - All 8 experts run simultaneously
- `use_fast_model=True` for Rounds 1-3 (Flash model)
- `use_fast_model=False` for Rounds 4-5 (Pro model)

**Parallelizable rounds**: 1, 2, 3, 4, 5 (all main rounds)
**Sequential rounds (disabled)**: 1.5, 3a-b-c, 4.5, synthesis steps

## Configurable Depth

| num_rounds | Output | Use Case |
|------------|--------|----------|
| 1 | Critique only (limitation map) | Quick assessment |
| 2-3 | + Brainstorm + Cross-pollination | Ideation focus |
| 5 | Full deliberation with voting & stress-test | Production quality |

## Quick Mode

Reduces expert panel and combines rounds:

```python
QUICK_REVIEW_CONFIG = {
    "rounds": 2,
    "round_1": "critique + propose (combined)",
    "round_2": "vote + stress test (combined)",
    "experts": ["Health Literacy Specialist", "Patient Advocate", "Medical Ethicist"],
    "time_budget": "2 minutes"
}
```

---

## Potential Applications

This architecture is domain-agnostic. The pattern works for any problem where:
1. Multiple perspectives add value
2. Disagreement should be surfaced, not hidden
3. Solutions need stress-testing before deployment
4. Quality matters more than speed

### 1. **Policy Review**
- Experts: Legal, Compliance, Operations, Customer Advocate
- Round 1: Find risks in proposed policy
- Round 3: Challenge each other's interpretations
- Round 5: Simulate customer edge cases

### 2. **Product Feature Design**
- Experts: UX, Engineering, Product, Accessibility, Security
- Round 1: Critique proposed feature
- Round 2: Brainstorm alternatives
- Round 4: Vote on implementation approach

### 3. **Legal Document Review**
- Experts: Contract Law, IP, Employment, Risk
- Round 1: Identify ambiguities and risks
- Round 3: Challenge interpretations
- Round 5: Simulate adversarial readings

### 4. **Investment Thesis Evaluation**
- Experts: Bull Analyst, Bear Analyst, Risk Manager, Industry Expert
- Round 1: Independent analysis
- Round 3: Challenge each other's assumptions
- Round 4: Converge on recommendation with dissents

### 5. **Code Review / Architecture Decision**
- Experts: Security, Performance, Maintainability, DX
- Round 1: Identify concerns independently
- Round 3: Debate trade-offs
- Round 4: Vote with dissenting views captured

### 6. **Marketing Copy / Brand Messaging**
- Experts: Brand, Legal, Cultural Sensitivity, Target Audience Rep
- Round 1: Critique proposed messaging
- Round 2: Generate alternatives
- Round 5: Simulate audience reactions

### 7. **Clinical Trial Protocol Review**
- Experts: Biostatistician, Ethicist, Clinician, Patient Advocate
- Round 1: Identify protocol risks
- Round 3: Challenge each other's concerns
- Round 5: Simulate edge cases

---

## Implementation Notes

### Scaling Considerations
- **max_workers**: Balance parallelism vs API rate limits
- **Model selection**: Use faster model for early rounds, reasoning model for synthesis
- **Caching**: Expert prompts are deterministic per round - can cache

### Quality vs Speed Trade-offs
| Setting | Time | Quality | Description |
|---------|------|---------|-------------|
| Quick mode | ~20-30s | Good | 2 rounds, 3 experts |
| **Current (Optimized)** | **~40-60s** | **High** | 5 rounds, 8 experts, Flash R1-3, Pro R4-5 |
| Full mode (if re-enabled) | ~3-5min | Highest | All sub-rounds enabled |

### Model Selection Strategy
| Round | Model | Reasoning |
|-------|-------|-----------|
| 1 (Critique) | Flash | Discovery doesn't need deep reasoning |
| 2 (Brainstorm) | Flash | Divergent ideation benefits from speed |
| 3 (Cross-pollinate) | Flash | Building on ideas is pattern matching |
| 4 (Converge) | Pro | Voting/synthesis needs nuanced judgment |
| 5 (Stress test) | Pro | Patient simulation needs empathy/depth |

### Key Design Decisions
1. **Critique before brainstorm**: Prevents anchoring on first ideas
2. **Explicit dissent capture**: Minority views are features, not bugs
3. **Challenge-response dialogue**: Forces experts to engage with each other
4. **Patient personas**: Ground-truths against real-world failure modes
5. **Lane enforcement**: Prevents expert homogenization

---

## File References

- **Service**: `services/terminology_roundtable_service.py`
- **UI Input**: `ui/roundtable_input.py`
- **UI Results**: `ui/roundtable_results.py`
- **Domain Experts**: `domains/medical_communication/__init__.py`

---

## Re-enabling Disabled Features

The following features were disabled for speed but can be re-enabled in `services/terminology_roundtable_service.py`:

### Round 1.5 (Meta-Review)
```python
# In run_roundtable(), around line 704, uncomment:
# if num_rounds >= 2:
#     yield {"stage": "round_start", "round": "1.5", ...}
#     meta_review = self._run_meta_review(round1_responses, input_data)
#     ...
```

### Round 3a-3c (Challenge-Response)
```python
# In run_roundtable(), around line 817, the entire block is commented:
# if num_rounds >= 4:
#     # Round 3a: Challenge
#     # Round 3b: Respond
#     # Round 3c: Blind spot check
```

### When to Re-enable
- **Meta-review**: When you need deduplicated, scored findings
- **Challenge-response**: When you want true expert debate (not just parallel views)
- **Blind spot check**: When safety/ethics is critical

---

## Future Enhancements

1. **Async execution**: True async instead of ThreadPoolExecutor
2. **Streaming results**: Show partial output as rounds complete
3. **Human-in-the-loop**: Allow user to inject feedback between rounds
4. **Memory**: Learn from past roundtable outcomes
5. **A/B testing**: Compare different expert panel compositions
6. **Adaptive depth**: Auto-enable sub-rounds based on topic complexity
