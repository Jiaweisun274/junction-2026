# References: what the environment is built on

For each part of the environment: the source, what we took from it, and which numbers are our own choices.
Code locations refer to `probe/tutor2.py` (the simulator) and `probe/lit_ref.py` (the literature policy).

## 1. Mechanisms of the simulated child

| mechanism | in the simulator | source | taken from the source | our own choice |
|---|---|---|---|---|
| Hidden mastery, noisy quizzes | each concept is mastered or not; a quiz is answered right with 0.9 if mastered (slip 0.10), 0.25 if not (guess) | Corbett & Anderson (1995) [1] | the hidden binary state with slip and guess parameters (Bayesian Knowledge Tracing) | the values 0.10 and 0.25; learn chance per explanation U(0.3, 0.6) |
| Forgetting | recall = 2^(−Δt / h); new concept h ~ U(2.6, 7.8) days, known before the course h ~ U(8, 20) | Ebbinghaus (1885) [2]; Settles & Meeder (2016) [3]; FSRS [4, 5] | exponential forgetting with a half-life [3]; half-life ranges converted from the FSRS-5 default first-review stabilities (0.40 / 1.18 / 3.17 days to 90% recall, × 6.58 to a half-life) [5] | uniform ranges between those defaults |
| Testing effect | a correct recall multiplies h by 2.5 × (2 − recall) | Roediger & Karpicke (2006) [6]; SM-2 [7] | retrieval strengthens memory more than restudy [6]; 2.5 is the SM-2 default ease factor [7] | the (2 − recall) bonus for recalling a fading memory |
| Spacing and restudy | re-explaining a known concept multiplies h by 1.5 × (2 − recall) | Cepeda et al. (2006) [8] | spaced review helps more than immediate review | the factor 1.5 |
| Massed practice | each earlier explanation of the same concept in the same session multiplies the learn chance by 0.6 | Cepeda et al. (2006) [8]; Dunlosky et al. (2013) [9] | massed repetition gives diminishing returns; a new session starts fresh | the factor 0.6 |
| Prerequisites | a concept is learned with chance 0.05 if its prerequisite is not mastered | Doignon & Falmagne (1985, 1999) [10, 11] | knowledge forms a prerequisite structure | a single chain; the leak 0.05 |
| Misconceptions | bigger denominator = bigger fraction; add to top and bottom; add across (carried into word problems); signature wrong answer with 0.9 | Brown & Burton (1978) [12]; Ni & Zhou (2005) [13]; Siegler et al. (2013) [14] | systematic "bugs" with diagnostic answers [12]; whole-number bias in fractions [13]; adding numerators and denominators is a documented common error [14] | susceptibility 0.4, formed with 0.6 on learning, targeted correction fixes 0.70, plain explanation 0.10 |
| Over-confident self-report | "do you get it?" answered yes with 0.75–0.97 regardless of knowledge | Rozenblit & Keil (2002) [15] | people overestimate how well they understand | all four probabilities |
| Exam | mastered: 0.9 × recall + 0.25 × (1 − recall); misconception 0.10; unlearned 0.25 | follows from the quiz model above | | misconception exam score 0.10 |

Calibration history: our first simulator used a 1-day half-life. A quick fit of half-life
regression [3] on the first 2M rows of the Duolingo data showed real forgetting is far slower, so we switched to the
FSRS-derived ranges [5]. A second realism fix added diminishing returns for massed explanations [8, 9].

## 2. Reference and comparison teachers

| teacher | built from | source |
|---|---|---|
| Literature rulebook (`probe/lit_ref.py`) | mastery learning with BKT (mastery at P ≥ 0.95), Leitner-style spaced review at 90% target retention, prerequisite back-off, targeted remediation | Bloom (1968) [16]; Corbett & Anderson (1995) [1]; Leitner (1972) [17]; Settles & Meeder (2016) [3]; Tabibian et al. (2019) [18]; Doignon & Falmagne (1999) [11]; Brown & Burton (1978) [12] |
| PPO reference (`probe/rl_ref.py`) | policy-gradient reinforcement learning on observation-only features | Schulman et al. (2017) [19] |
| Planned 5-rule teacher (`probe/planned_ref.py`) | mastery criterion of two correct in a row, prerequisite check, diagnosis | mastery learning [16]; real tutoring systems also require several correct answers in a row before moving on, e.g. ASSISTments skill builders [20] |
| Full-information teacher (normalisation only) | greedy and rollout planning with the hidden state visible | teaching as a POMDP: Rafferty et al. (2016) [21] |

## 3. Gemma tiebreaker (training a small model on the environment's experience)

Reward-filtered self-training (expert iteration): sample episodes, keep each child's best by exam gain, fine-tune on
them, repeat. Sources: Anthony et al. (2017) [22]; Zelikman et al. (2022) [23]; Gulcehre et al. (2023) [24].

## 4. Data for future calibration

ASSISTments tutoring logs [20, 25]; Eedi diagnostic questions (NeurIPS 2020 Education Challenge) [26];
Duolingo spaced-repetition data [3].

## Bibliography

1. Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278.
2. Ebbinghaus, H. (1885). *Über das Gedächtnis*. Leipzig: Duncker & Humblot.
3. Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. *Proceedings of ACL 2016*, 1848–1858.
4. Ye, J., Su, J., & Cao, Y. (2022). A stochastic shortest path algorithm for optimizing spaced repetition scheduling. *Proceedings of KDD 2022*.
5. Open Spaced Repetition. FSRS-5 algorithm and default parameters (software; used by Anki). github.com/open-spaced-repetition
6. Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science*, 17(3), 249–255.
7. Woźniak, P. A. (1990). *Optimization of learning* (SuperMemo SM-2 algorithm). Master's thesis, University of Technology in Poznań.
8. Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354–380.
9. Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques. *Psychological Science in the Public Interest*, 14(1), 4–58.
10. Doignon, J.-P., & Falmagne, J.-C. (1985). Spaces for the assessment of knowledge. *International Journal of Man-Machine Studies*, 23(2), 175–196.
11. Doignon, J.-P., & Falmagne, J.-C. (1999). *Knowledge Spaces*. Springer.
12. Brown, J. S., & Burton, R. R. (1978). Diagnostic models for procedural bugs in basic mathematical skills. *Cognitive Science*, 2(2), 155–192.
13. Ni, Y., & Zhou, Y.-D. (2005). Teaching and learning fraction and rational numbers: The origins and implications of whole number bias. *Educational Psychologist*, 40(1), 27–52.
14. Siegler, R. S., Fazio, L. K., Bailey, D. H., & Zhou, X. (2013). Fractions: The new frontier for theories of numerical development. *Trends in Cognitive Sciences*, 17(1), 13–19.
15. Rozenblit, L., & Keil, F. (2002). The misunderstood limits of folk science: An illusion of explanatory depth. *Cognitive Science*, 26(5), 521–562.
16. Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment*, 1(2).
17. Leitner, S. (1972). *So lernt man lernen*. Freiburg: Herder.
18. Tabibian, B., Upadhyay, U., De, A., Zarezade, A., Schölkopf, B., & Gomez-Rodriguez, M. (2019). Enhancing human learning via spaced repetition optimization. *PNAS*, 116(10), 3988–3993.
19. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
20. Heffernan, N. T., & Heffernan, C. L. (2014). The ASSISTments ecosystem. *International Journal of Artificial Intelligence in Education*, 24(4), 470–497.
21. Rafferty, A. N., Brunskill, E., Griffiths, T. L., & Shafto, P. (2016). Faster teaching via POMDP planning. *Cognitive Science*, 40(6), 1290–1332.
22. Anthony, T., Tian, Z., & Barber, D. (2017). Thinking fast and slow with deep learning and tree search. *NeurIPS 2017*.
23. Zelikman, E., Wu, Y., Mu, J., & Goodman, N. (2022). STaR: Bootstrapping reasoning with reasoning. *NeurIPS 2022*.
24. Gulcehre, C., et al. (2023). Reinforced self-training (ReST) for language modeling. arXiv:2308.08998.
25. Feng, M., Heffernan, N., & Koedinger, K. (2009). Addressing the assessment challenge with an online system that tutors as it assesses. *User Modeling and User-Adapted Interaction*, 19(3), 243–266.
26. Wang, Z., et al. (2020). Instructions and guide for diagnostic questions: The NeurIPS 2020 Education Challenge. arXiv:2007.12061.

Citations were compiled from memory during the event; check volume and page numbers before reusing them in a paper.
