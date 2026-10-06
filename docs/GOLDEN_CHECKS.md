# Golden set: external cross-checks

The gold SQL is verified by execution and by internal consistency tests, but a query can run
cleanly and still be wrong. These checks compare gold answers with sources outside this project.
The plan promised 15; **3 checks covering 5 gold facts are done**. The rest are open work.

| Gold fact | Question(s) | Gold value | Independent source | Result |
|---|---|---|---|---|
| Largest single-game scoring total | L08, J09 | Bam Adebayo, 83 points, 2026-03-10 | Multiple news reports ([SI](https://www.si.com/nba/bam-adebayo-eclipses-kobe-bryant-83-points-second-wilt-chamberlain-heat-wizards), [NPR/VPM](https://www.vpm.org/npr-news/2026-03-11/bam-heats-adebayo-scores-83-points-2nd-only-to-wilt-chamberlain-in-nba-history)) | matches |
| Season win totals | L18, M03, J01 | OKC 64, SAS 62, DET 60, BOS 56 | [Champs or Chumps final standings](https://champsorchumps.us/summary/nba/2026) (via search summary) | matches |
| Largest margin of victory | L17 | UTA vs CHA, 55 points (Charlotte 150, Utah 95, 2026-01-10) | [ESPN recap](https://www.espn.com/nba/recap/_/gameId/401810402), [Deseret News](https://www.deseret.com/sports/2026/01/10/utah-jazz-worst-loss-charlotte-hornets/) | matches |

## Still to check (suggested)

- Season totals for a few players against Basketball-Reference (L01 Tatum 348 points, L16 Curry 190 threes, A08 Wembanyama 197 blocks, A09 Wallace 150 steals).
- A few contract figures against Spotrac or Basketball-Reference (L05 Curry $62,587,158, T14 Beal $25,807,810).
- 2026-01-15 game count (L09) and Christmas Day results (L02) against the NBA schedule.

## What these checks do and do not show

They show the database agrees with published facts on the cases above, so a model that answers
correctly from the database is also correct in the world. They do not validate the *derived* metrics
(`shrunk`, RAPM, `distinguishable`); those are only as good as the source project's model, which is
documented in its DESIGN.md and explicitly descriptive.
