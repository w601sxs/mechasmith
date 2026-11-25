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

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| The language expresses real robots faithfully | Table 1 (generated): 6 reference robots, axes within 0.8 mm, mass within 6.2 % | have it; add per-axis errors from `compare_joints.py` |
| Models are usable for learning unedited | `examples/rl_microduck`, `examples/rl_jetpack` | have runs; need curves, seeds, hardware |
| Agents build valid robots more often with mechasmith | Q2 agent study | **not run** |
| Each verifier contributes | Q2 ablations | **not run** |

The agent study is the paper's ML contribution and the main risk. Without it this is a systems
paper, which fits ICML poorly.

## Q2 protocol (to do)

1. **Task set.** Rebuild (reference numbers, no model file), Specify (natural-language brief with a
   class-test target), Repair (designs with injected faults). Aim for 30+ tasks; hold out robots
   not among the bundled examples so the examples cannot be copied.
2. **Conditions.** (A) raw MJCF/URDF; (B) build123d with free placement and an MJCF exporter;
   (C) mechasmith; (C-) ablations: no contact check, no sweep, no class test, no parameter ranges.
   Same agent, prompt scaffold, tool budget and wall-clock limit everywhere.
3. **Checker.** One independent scorer for all conditions: load, connectivity, home collisions,
   sweep, axis error to reference, mass error, class test. Write it first; it is also the
   reviewers' main question.
4. **Runs.** At least 2 agent models, 3+ seeds per task. Report mean and bootstrap CIs.
5. **Leakage.** Agents must not see the reference model files; log tool calls to check.

## Next steps

- [ ] Write the independent checker and the fault injector.
- [ ] Assemble the Q2 task set (references not yet in `designs/`).
- [ ] Run a pilot (3 tasks x 3 conditions) to size the budget.
- [ ] Axis-error columns in `collect_results.py`.
- [ ] Figure 2: snippet, `AttachmentError`, report fragment.
- [ ] Formal statement of the connectivity invariant.
- [ ] RL curves for Q3.
- [ ] Verify all bibliography entries.
