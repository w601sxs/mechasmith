# Contributing to mechasmith

Thanks for looking. The most useful contribution is a robot: a design file for something your lab
builds or trains on. Every new design shows what the library can't express yet, and that is how
the library gets better.

## Ways to help

- **Add a robot.** Open an issue with the robot and its public reference (a MuJoCo Menagerie
  model, a vendor URDF, a paper's numbers), or send the design. Issues labelled
  [`new robot`](https://github.com/w601sxs/mechasmith/labels/new%20robot) are ones we'd like.
- **Report what broke.** A design that won't build, a report number that looks wrong, a joint
  axis that doesn't match the reference. Include the design file and `report.json`.
- **Add a library feature** a design needed. Keep the design simple and put the capability in the
  library (see [AGENTS.md](AGENTS.md), "Working on the library itself").
- **Tell us you use it.** Comment on the
  [Who's using mechasmith?](https://github.com/w601sxs/mechasmith/issues?q=is%3Aissue+%22Who%27s+using+mechasmith%22)
  issue. Say whether we may list your lab or company in the README.

## Setup

```bash
git clone https://github.com/w601sxs/mechasmith && cd mechasmith
uv sync
uv run mechasmith build go2        # any bundled design; writes build/go2/
```

## Adding a robot

1. Start from the closest example: `uv run mechasmith copy <example> designs/myrobot.py`.
   [AGENTS.md](AGENTS.md) has a table of which example to start from.
2. Match the reference's joint axes first (`scripts/compare_joints.py`), then the masses per
   body, then the look. [docs/iterations/](docs/iterations/) shows how this went for the Robotiq,
   Crazyflie, Go2 and G1.
3. Follow the rules in [AGENTS.md](AGENTS.md): parts enter only through connections, every
   parameter is declared with `param(...)`, build one side and mirror it.
4. Add the design to the CI matrix in `.github/workflows/ci.yml`.

## Before you open a pull request

- Build every design: `for d in designs/*.py; do uv run mechasmith build $d --no-sweep > /dev/null || echo "FAILED $d"; done`.
  Masses, collisions and the stand, hover and arm results should only move if you meant them to.
  Say so in the PR when they do.
- Keep pull requests to one change. A new robot and a library feature it needs can go together.

Licensing: contributions are under the [MIT license](LICENSE). Only add a reference model's
geometry or numbers if its license allows it, and credit it in the design's docstring.
