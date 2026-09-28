"""anima3 — a thin System-One brain for Ultima Online, driving anima-client directly.

The body (anima-client's `anima-bridge` NDJSON bridge) owns the wire. This brain
renders each Observation into a short text *scene*, enumerates the verbs that are
valid right now (a closed vocabulary with hard safety limits baked in), and asks a
decision model to pick one by reading the log-probabilities of the option letters —
never by generating text. Every pick carries a probability distribution, so a low
confidence falls back to the first (rule-ordered) verb. Every decision is logged.
"""
