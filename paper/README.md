# mechasmith: ICML 2027 paper

Working title: *Connect, Don't Place: A Verifiable Robot Description Language for Coding Agents*.

## Build

```bash
sudo apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended latexmk
make            # main.pdf
make results    # rebuild every design and regenerate generated/*.tex (run from paper/)
make check      # fails while any \planned{...} or \note{...} is left in the text
```

On a headless machine `make results` needs an X server for the renders:
`MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x1024x24" make results`.

**Style files.** ICML 2027's author kit is not published yet, so the ICML 2026 files are vendored
here as a stand-in. When the 2027 kit appears, add `icml2027.sty`/`icml2027.bst`, change the two
lines marked `STYLE` in `main.tex`, and re-read the 2027 call for papers (page limit, impact
statement, checklist) before submitting. Check the official deadline on icml.cc.

## Layout

| Path | What |
|---|---|
| `main.tex` | preamble, title, section includes |
| `sections/*.tex` | one file per section |
| `generated/` | tables written by `scripts/collect_results.py`. Never edit by hand |
| `figures/` | figures (`gallery.png` is a copy of `docs/media/gallery.png`) |
| `refs.bib` | bibliography. Verify every entry against the publisher's page before submission |

## Thesis

Coding agents fail at robot modelling because MJCF/URDF encode every body as a free pose, so
errors are silent. A language where parts can only enter through checked connections, with every
frame derived and every property reported as JSON, turns robot design into a task agents can
iterate on against verifiers.

## Positioning (literature survey, 2026-11-25)

Verdict: the paper is valid, but the novelty is the combination, not any single idea.

| Prior work | Already does | Does not do |
|---|---|---|
| Articraft (NeurIPS 2026) | agent writes Python against an SDK; harness reports floating/overlapping parts; SDK preferred over raw URDF in a user study | actuators, coordinate-free placement, joint sweeps, dynamics, MJCF; objects only |
| RoboForge (arXiv 2610.17881) | MLLM generates actuated servo robots (STEP/STL/URDF), scored by post-hoc checks | checks in the agent loop (humans fix failures); enforced connections; code not released |
| ArtiCAD, AssemCAD (arXiv 2026) | connector/mate-based placement of LLM-generated CAD assemblies | actuators, mass, simulation |
| Articulate-Anything, Real2Code, URDF-Anything | articulation as code / URDF prediction | actuated robots, dynamics checks |
| RoboMorph, Text2Robot, Debate2Create | LLM robot design inside templates | open-ended design language |
| Exarchos 2021, Campanaro 2024 | smooth kinematic/dynamic perturbation of policies | discrete authoring faults evaluated on the correct model |

Do **not** claim: first to have agents write articulated assets as code, first with verifier
feedback, first connection-based placement, or first evidence that an SDK beats raw URDF.

Claim: actuated robots built only through checked connections with every frame derived; joint
sweeps and stand/hover/payload tests in the agent loop; the first controlled, objectively scored
interface benchmark for actuated robots (extending Articraft's SDK-vs-URDF finding); the first
study of discrete authoring faults on learned policies.

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| The language expresses real robots faithfully | Table 1 (generated): 6 reference robots, mass within 6.2 % | have it; add axis errors (Articulate-Anything definitions) |
| Agents build valid robots more often with mechasmith | Q2 benchmark | **not run** |
| Feedback alone does not explain the gain | Q2 condition A+ (raw URDF + checker) | **not run** |
| Authoring faults corrupt policies silently | Q3 fault injection | **not run** |
| Models are usable for learning unedited | `examples/rl_microduck`, `examples/rl_jetpack` | have runs; need curves, seeds |

## Q2 protocol (to do)

1. **Tasks.** Rebuild (reference numbers, no model file), Specify (brief with a class-test target),
   Repair (injected faults). 30+ tasks on robots not in `designs/`.
2. **Conditions.** (A) raw MJCF/URDF; (A+) raw plus the checker as a tool; (B) CadQuery with render
   feedback (CADCodeVerify style) and an MJCF exporter; (C) Articraft SDK, URDF to MJCF, actuators
   added from the spec; (D) mechasmith; (D-) ablations. Same model, scaffold, budget everywhere.
3. **Checker.** Reads only MJCF: loads, floating bodies, home collisions, axis/origin error
   (Articulate-Anything: success below 0.25 rad and 50 mm), mass error, class test.
4. **Runs.** 2+ agent models, 3+ seeds per task, bootstrap CIs.
5. **Leakage.** No reference model files visible; log tool calls.

## Q3 protocol (to do)

Go2 and Microduck; faults: shifted part, flipped/tilted axis, interpenetration, mass/inertia
scale; 3 severities; 3 seeds; PPO on faulty model, evaluate on correct model; plus a
domain-randomised policy per fault. About 90 runs (~40 GPU-hours if Go2 trains as fast as
Microduck; unmeasured).

## Next steps

- [ ] Write the independent checker.
- [ ] Assemble the Q2 task set (references not yet in `designs/`).
- [ ] Run a pilot (3 tasks x 3 conditions) to size the budget.
- [ ] Axis-error columns in `collect_results.py`.
- [ ] Figure 2: snippet, `AttachmentError`, report fragment.
- [ ] Formal statement of the connectivity invariant.
- [ ] RL curves and seeds for Q4.
- [ ] Verify the 14 bibliography entries above the VERIFIED line in `refs.bib`.
- [ ] Confirm Articraft's NeurIPS 2026 venue (one source only) and Debate2Create's venue.
- [ ] Write the fault injector and start Q3 runs (independent of Q2).
