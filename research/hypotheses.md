# Hypotheses

Executable one-axis hypotheses live in `research/search_space.py`:

- `night_one`: scratch byte-transformer topology, pooling, head, and loss ablations.
- `jevlet_p`: the same axes on a pretrained encoder, plus exact order invariance
  (`option_isolated`), sibling leakage (`full`), and backbone choice.

Queued, not yet executable:

1. **Teacher soft labels (RLCD proxy).** Average option probabilities from two or more
   frontier models over generated daily-driver states and train on them with a proper
   scoring rule. Expected: better calibration and better transfer to real phrasing than
   template labels. Needs an API budget decision.
2. **Two-stage choice beyond 255 options** with `option_isolated` as the independent scorer:
   options share positions, so the scorer's cost grows with packed length, not position.
3. **State-prefix caching.** State tokens never attend to branches, so a state's encoding can
   be reused across calls that add questions. Measure latency for repeated screen states.
4. **Screen grounding.** A Choice over the visible UI Automation controls ("which control
   accomplishes this task?"), trained from demonstrations the user performs.
5. Latent compression, token pruning, adaptive depth, sparse experts (unchanged from night one).
