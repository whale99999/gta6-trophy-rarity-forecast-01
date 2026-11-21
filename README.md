# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/whale99999/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/whale99999/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Dilution-aware forecaster for GTA VI trophy earn rates on Exophase. A trophy's earn rate is a ratio — players who earned it over players tracked — and Exophase only moves the page when it refreshes: each refresh jumps the tracked-player count (launch night: 638 → 3,868) while the newly tracked players hold far fewer trophies, so every rate except the universal one falls ~5x. This program forecasts the ratio, not the rate: it fits ln(players-earned) and ln(players-tracked) as separate trends on ln(1 + hours since release) over the distinct refresh points, shrinks each trophy's slope toward the pooled slope, anchors at the last read, and reads both at the target — so refresh dilution and player progression are both captured. With a flat history (the practice cases) it uses the launch-night refresh prior: median 0.20x the last rate, logit spread 0.70. Spreads are honest and wide, calibrated on the practice window and two synthetic final-like worlds. Practice pinball loss ~0.17 vs the house baseline's ~1.51.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=881py9z1&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
