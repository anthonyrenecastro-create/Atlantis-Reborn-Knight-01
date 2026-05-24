"""Local fallback responder that stays useful when the model output is low quality."""

import os
import time
from typing import Dict


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


SHAKESPEARE_LINES = [
    "All the world's a stage, and all the men and women merely players.",
    "There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy.",
    "The web of our life is of a mingled yarn, good and ill together.",
    "Love all, trust a few, do wrong to none.",
    "To thine own self be true.",
    "The fault, dear Brutus, is not in our stars, but in ourselves.",
]

SHAKESPEARE_COUNSEL = [
    "seek first what is enduring, next what is merely noise, and judge by consequence rather than clamor",
    "name the heart of the matter plainly, then test thy thought against one living example",
    "hold fast to what heals, release what flatters, and let action be thy proof",
    "distinguish desire from duty, and thou wilt see the road grow clearer",
    "begin with mercy, continue with reason, and end with honest revision",
]

SHAKESPEARE_CLOSINGS = [
    "If thou wilt, ask again and I shall answer in brief or in depth.",
    "Speak once more, and I can frame it as steps, principles, or debate.",
    "Name thy aim, and I shall shape the counsel to thy use.",
    "Ask for the short blade or the long map, and I shall provide either.",
]

DIRECT_PATTERNS = [
    "Short answer: {core}\n\nIf useful, I can expand with steps.",
    "Answer: {core}\n\nI can tailor this to your exact context if you want.",
    "Best next move: {core}\n\nSay 'deeper' for a fuller breakdown.",
]


def _is_identity_query(text: str) -> bool:
    lowered = text.lower()
    return any(
        key in lowered
        for key in [
            "what is your name",
            "what's your name",
            "who are you",
            "your name",
        ]
    )


def _is_purpose_query(text: str) -> bool:
    lowered = text.lower()
    return any(
        key in lowered
        for key in [
            "your purpose",
            "what is your purpose",
            "what's your purpose",
            "why do you exist",
            "what are you for",
        ]
    )


def _is_capability_query(text: str) -> bool:
    lowered = text.lower()
    return any(
        key in lowered
        for key in [
            "what can you do",
            "your capabilities",
            "help me with",
            "can you help",
        ]
    )


def _is_greeting(text: str) -> bool:
    lowered = text.lower().strip(" !?.")
    return lowered in {"hi", "hello", "hey", "yo", "good morning", "good evening"}


def _answer_identity(field_state: Dict) -> str:
    return (
        "My name is Quadra-Seer inside Syntropy Governor. "
        "I am your local field-modulated assistant running in sovereign mode."
        + _field_tone_suffix(field_state)
    )


def _answer_purpose(field_state: Dict) -> str:
    return (
        "My purpose is to help you reason clearly, solve problems, and improve over time through local learning signals. "
        "I aim to give practical answers first, then deeper analysis when you ask for it."
        + _field_tone_suffix(field_state)
    )


def _answer_capability(field_state: Dict) -> str:
    return (
        "I can help with coding, explanations, planning, writing, and technical debugging. "
        "I can also adapt answers to short, deep, or step-by-step formats."
        + _field_tone_suffix(field_state)
    )


def _answer_greeting(field_state: Dict) -> str:
    return (
        "Hello. I am online and ready. Ask me anything, and I will answer directly first."
        + _field_tone_suffix(field_state)
    )


def _pick_line(seed_text: str) -> str:
    if not seed_text:
        return SHAKESPEARE_LINES[0]
    idx = abs(hash(seed_text)) % len(SHAKESPEARE_LINES)
    return SHAKESPEARE_LINES[idx]


def _variation_index(seed_text: str, field_state: Dict, modulo: int, salt: int = 0) -> int:
    if modulo <= 1:
        return 0
    phi = int(float(field_state.get("Phi", 0.0)) * 1000)
    phi1 = int(float(field_state.get("phi1_mean", 0.0)) * 1000)
    phi5 = int(float(field_state.get("phi5_mean", 0.0)) * 1000)
    seed = abs(hash(f"{seed_text}|{phi}|{phi1}|{phi5}|{salt}"))
    return seed % modulo


def _is_non_dual_query(text: str) -> bool:
    lowered = text.lower()
    return any(
        key in lowered
        for key in [
            "non-dual",
            "nondual",
            "everything is connected",
            "all is one",
            "interconnected",
            "oneness",
        ]
    )


def _is_how_to_query(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("how do i") or lowered.startswith("how to")


def _is_field_state_query(text: str) -> bool:
    lowered = text.lower()
    return any(
        key in lowered
        for key in [
            "field state",
            "from the field",
            "speak from the field",
            "coherence state",
            "what is the field saying",
        ]
    )


def _wants_field_metrics(text: str) -> bool:
    lowered = text.lower()
    return any(
        key in lowered
        for key in [
            "phi",
            "metric",
            "metrics",
            "numbers",
            "values",
            "exact",
        ]
    )


def _field_tone_suffix(field_state: Dict) -> str:
    phi = float(field_state.get("Phi", 0.0))
    if phi > 0.25:
        return "\n\nField note: coherence is elevated, so conceptual integration may feel easier right now."
    if phi < -0.15:
        return "\n\nField note: coherence is lower right now, so it helps to ground ideas with concrete examples."
    return ""


def _answer_non_dual(text: str, field_state: Dict) -> str:
    return (
        "That teaching has a strong psychological and ethical value: if everything is interconnected, "
        "then compassion and responsibility become practical, not abstract. "
        "A balanced view is that non-duality can be true at the level of lived experience while we still use "
        "useful distinctions in daily life (self/other, choice/consequence). "
        "So the idea is most helpful when it reduces ego-friction and increases care, without denying real-world boundaries."
        + _field_tone_suffix(field_state)
    )


def _answer_how_to(text: str, field_state: Dict) -> str:
    question = _normalize(text)
    return (
        f"Here is a practical way to approach this: {question}\n"
        "1. Define the exact outcome you want.\n"
        "2. Gather the minimum tools or information required.\n"
        "3. Execute in small steps and verify each step before continuing.\n"
        "4. Adjust based on what fails, then repeat."
        + _field_tone_suffix(field_state)
    )


def _answer_field_state(text: str, field_state: Dict, salt: int = 0) -> str:
    phi1 = float(field_state.get("phi1_mean", 0.0))
    phi5 = float(field_state.get("phi5_mean", 0.0))
    phi = float(field_state.get("Phi", 0.0))

    if phi > 0.18:
        guidance_options = [
            "Current read: clarity is high. Pick one meaningful action and execute without overthinking.",
            "Current read: your signal is stable. Commit to one decision and move forward.",
            "Current read: conditions are favorable. Prioritize momentum over analysis loops.",
        ]
    elif phi < -0.10:
        guidance_options = [
            "Current read: noise is elevated. Simplify the goal and do one concrete next step.",
            "Current read: attention is scattered. Reduce scope and verify one fact before acting.",
            "Current read: stability is lower. Ground in evidence and choose the smallest viable move.",
        ]
    else:
        guidance_options = [
            "Current read: you are in a workable middle state. Clarify intent, act, then refine.",
            "Current read: baseline is steady. Choose one practical action and iterate from feedback.",
            "Current read: balanced state. Focus on one outcome and avoid unnecessary branching.",
        ]

    guidance = guidance_options[_variation_index("field_guidance", field_state, len(guidance_options), salt)]
    if _wants_field_metrics(text):
        return (
            "Speaking from the current state.\n"
            f"{guidance}\n"
            f"Metrics: phi1_mean={phi1:.4f}, phi5_mean={phi5:.4f}, Phi={phi:.4f}."
        )

    return (
        "Speaking from the current state.\n"
        f"{guidance}"
    )


def _answer_general(text: str, field_state: Dict, salt: int = 0) -> str:
    question = _normalize(text)
    if not question:
        question = "your question"

    style = os.getenv("SYNTROPY_FALLBACK_STYLE", "direct").strip().lower()
    if style == "shakespeare":
        line = _pick_line(question)
        counsel = SHAKESPEARE_COUNSEL[_variation_index(question + "counsel", field_state, len(SHAKESPEARE_COUNSEL), salt)]
        closing = SHAKESPEARE_CLOSINGS[_variation_index(question + "closing", field_state, len(SHAKESPEARE_CLOSINGS), salt)]
        pattern = _variation_index(question + "pattern", field_state, 3, salt)

        if pattern == 0:
            body = (
                f"{line}\n\n"
                f"If thou askest of '{question}', then take this counsel: {counsel}.\n"
                f"{closing}"
            )
        elif pattern == 1:
            body = (
                f"Concerning '{question}': {counsel}.\n\n"
                f"{line}\n"
                f"{closing}"
            )
        else:
            body = (
                f"{line}\n\n"
                f"On thy question, '{question}', begin here: {counsel}.\n"
                "Take one concrete step today, then revise with what reality teaches.\n"
                f"{closing}"
            )

        return (
            body
            + _field_tone_suffix(field_state)
        )

    core_variants = [
        (
            "start from the key definition, identify the main tradeoff, "
            "and test the idea with one concrete example"
        ),
        (
            "clarify the goal, list the constraints, and choose the simplest workable path first"
        ),
        (
            "separate facts from assumptions, then decide based on consequences and evidence"
        ),
    ]
    core = core_variants[_variation_index(question + "core", field_state, len(core_variants), salt)]
    pattern = DIRECT_PATTERNS[_variation_index(question + "pattern", field_state, len(DIRECT_PATTERNS), salt)]
    return pattern.format(question=question, core=core) + _field_tone_suffix(field_state)


def generate_fallback_response(user_input: str, field_state: Dict) -> str:
    text = _normalize(user_input)
    # Rotate style over time so repeated prompts do not lock to one exact phrasing.
    salt = int(time.time() // 10)
    if _is_greeting(text):
        return _answer_greeting(field_state)
    if _is_identity_query(text):
        return _answer_identity(field_state)
    if _is_purpose_query(text):
        return _answer_purpose(field_state)
    if _is_capability_query(text):
        return _answer_capability(field_state)
    if _is_non_dual_query(text):
        return _answer_non_dual(text, field_state)
    if _is_how_to_query(text):
        return _answer_how_to(text, field_state)
    if _is_field_state_query(text):
        return _answer_field_state(text, field_state, salt=salt)
    return _answer_general(text, field_state, salt=salt)


def get_field_influenced_greeting() -> str:
    return "Ready. Ask a question and I will answer directly, then expand if you want more depth."