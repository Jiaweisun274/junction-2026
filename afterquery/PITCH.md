# Pitch materials

- Simple demo (main demo): `viz/simple.html` ("Teach Aino"). Default child: Aino (class 3002, child 6).
- Full game (for curious judges): `viz/game.html` ("Tutor for a Week").
- Slides: "The Hidden-State Classroom" deck (speaker notes on every slide).
- Sources: `REFERENCES.md`. Lab notebook: `probe/RESULTS.md`.

Before recording or presenting: open the simple demo fresh (reload), x-ray off, step 1, child Aino.

## 2-minute video (about 280 words)

Screen recording of the simple demo with two or three slides cut in. Times are cumulative.

| time | on screen | voice-over |
|---|---|---|
| 0:00 | Simple demo, title | Can an AI teach a child better than a simple trained policy? We built an environment to find out. |
| 0:08 | Step 1: Teach idea 3, then Quiz 3; the child's answer in the bubble | You tutor a simulated child in fractions: 30 turns over a week, exam next Monday. You only see the answers. One right answer: is she ready to move on? |
| 0:25 | Turn on the x-ray | The x-ray shows what you can't see: what she really knows, how fast she forgets, which misconceptions she holds. Every rule is a documented effect from learning science: knowledge tracing, forgetting curves, the testing effect. |
| 0:42 | Step 2: Gemini vs reference, press Play | Here Gemini 3.1 Pro and our reference teach the same child. Purple is a quiz, teal is teaching. The AI keeps quizzing, and moves on after lucky guesses. |
| 0:58 | End of the week: 64% vs 81% | The reference teaches all six ideas; the AI two. |
| 1:05 | Slide: the gap (39 of 43) | Across five frontier models, the reference wins 39 of 43 runs. |
| 1:15 | Slide: four checks | Is it our fault? We gave the agent every parameter: no gain. We told it the right principles: its behaviour didn't change. We removed the teaching story: it got worse, so its real-world knowledge helps. And the result holds across 27 parameter settings. |
| 1:40 | Slide: five rules | Five rules a person could write down already beat the agents. The agents have the information and the principles; they don't act on them. |
| 1:52 | Slide: one command | Everything reruns with one command through Harbor. The Hidden-State Classroom, team Sauna Renaissance. |

Recording tips: Cmd+Shift+5 → record selected portion. Read the script aloud once with a timer first; if it runs
over 2:00, cut words rather than speeding up. In step 2 press Play, not Next turn, so it runs on its own.

## 10-minute live demo

| min | what | where |
|---|---|---|
| 0:00–0:30 | The question: can you teach better than an AI? | simple demo, title |
| 0:30–2:00 | Step 1, x-ray off. Teach idea 3, quiz it. If Aino answers right, ask the room: move on? Then turn the x-ray on and read the red line (the effect and its source). If the quiz is wrong, teach again and quiz once more. | simple demo |
| 2:00–3:00 | How the simulator is built: mechanism, source, which numbers are ours, calibration against Duolingo data | slide 4 |
| 3:00–4:30 | Step 2: Gemini 3.1 Pro vs reference on Aino, press Play. Point at the colour strip and the lucky-guess marks. End on 64% vs 81%. Say the AI wins on some children (switch to Toivo, class 3004: 82% vs 56%) and the gap is an average. | simple demo |
| 4:30–5:00 | Reward integrity: do-nothing 0, stop-early 0, state in a separate container | slide 5 |
| 5:00–5:30 | The gap: 39 of 43 runs | slide 6 |
| 5:30–7:00 | Four checks (information, principles, story, parameters) | slide 7 |
| 7:00–8:00 | Three decisions: testing instead of teaching, lucky guesses, damage done on Wednesday | slides 8–10 |
| 8:00–9:00 | Five rules beat the agents; Gemma tiebreaker; limits | slides 11–13 |
| 9:00–10:00 | One command; questions | slide 14 |

If short on time, skip slides 8–10 (the demo already showed the habits) and slide 12 (Gemma).

For the judges' rerun: `./eval.sh <model> 3001 3002 3003` (see README). A full three-seed run takes about an hour
with a Gemini model; have the seed-3002 result table ready on screen.

## Key numbers (check before saying them)

| claim | number | where |
|---|---|---|
| Reference wins | 39 of 43 frontier-agent runs | RESULTS.md, Harbor runs on 3001–3015 |
| Mean scores, 3001–3005 | PPO 0.70; Opus 0.59, GPT 0.61, Gemini 0.52–0.55; rulebook 0.56 | README |
| Baselines | do-nothing 0.00, stop-early 0.00, Monday-only 0.07 | README |
| Parameters disclosed | 0.52 → 0.56 and 0.55 → 0.52; reference wins 9/10 | RESULTS.md |
| Advice in the prompt | quiz share 66 → 64% and 65 → 67%; score +0.02 / −0.05 (15 batches) | RESULTS.md |
| Abstract skin | both models −0.065 (15 batches); reference wins 30/30 | RESULTS.md |
| Sensitivity | agent-like habit loses in 27/27 worlds | FAILURE_ANALYSIS.md |
| Agent habits | 61–68% quizzes vs PPO 37%; last 5 turns 61–77% vs 7%; lucky-guess move-ons 1.4–2.0 vs 0.5 per child | FAILURE_ANALYSIS.md |
| Five-rule teacher | 0.67 on 200 fresh batches (PPO 0.73, rulebook 0.56); beats Gemini 18/20 paired | RESULTS.md |
| Gemma 4 E4B | 0.37 → 0.39 → 0.41 over two rounds; +0.04 paired, se 0.04 | RESULTS.md |

## Questions to expect

- **Is the reference fair?** PPO sees only what the agent sees (its own actions and the child's answers), through
  the same MCP tools. It was trained on seeds outside the evaluation range; evaluation children come from salted seeds.
  A five-rule teacher a person could write down also beats the agents.
- **Isn't the simulator unrealistic?** Every mechanism is a documented effect; forgetting uses FSRS parameters fitted
  on real reviews; some numbers are ours and we say which. Removing the teaching story made the agents worse, so their
  real-world knowledge helps rather than misleads them. The result holds across 27 parameter settings. It is not
  fitted to real student data; ASSISTments or Eedi logs are the next step.
- **Maybe the agent lacks information?** We gave it every parameter: no improvement.
- **Maybe it doesn't know what to do?** We told it the key principles in the prompt: its quiz share did not change.
- **Can the agent game the reward?** Do-nothing and stop-early score 0. The state lives in a separate container, a
  separate verifier scores it, and the two exploits an independent review found are fixed.
- **Which models were the checks run on?** Two Gemini models (API budget). The main gap was measured on five models
  from three labs.
- **Could a smaller model learn this?** PPO is itself a small network trained only on the environment's experience.
  Gemma 4 E4B, trained on its own reward-filtered episodes, improved from 0.37 to 0.41 in two rounds: a trend, not yet
  significant.
- **Why does the AI win on some children?** The gap is an average. Each child has different learn rates and luck;
  on a few children the agent's habits happen to work.
