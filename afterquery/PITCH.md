# Pitch materials

Slides: "The Hidden-State Classroom" deck (Claude artifact; speaker notes on every slide).
Live demo page: `viz/index.html` (Hidden-State Classroom).

## 2-minute video (about 280 words)

Record as a screen capture of the slides plus a short clip of the demo page. Times are cumulative.

| time | on screen | voice-over |
|---|---|---|
| 0:00 | Cover | Can a frontier AI agent tutor a child? We built an environment to find out, and the answer is: about as well as a textbook, and clearly worse than a small policy trained in the environment. |
| 0:15 | Slide 2, the task | The agent tutors six simulated children in fractions. Each child gets three short sessions in a week. Every turn it can explain, quiz, correct a misconception, or ask “do you get it?”. The score is next Monday’s exam. |
| 0:35 | Slide 3, hidden state | What the agent cannot see is the child’s mind: what they know, how fast they forget, which misconceptions they hold. Every rule comes from learning science: knowledge tracing, forgetting curves, the testing effect. |
| 0:55 | Slide 5, results | Five frontier models from three labs score between 0.52 and 0.61. A small trained policy, seeing exactly what the agent sees, scores 0.70 and wins 20 of 23 runs. Doing nothing scores zero. |
| 1:10 | Slide 6, disclosed | Is the agent missing information? We told it every parameter of the simulator. The gap stayed. |
| 1:20 | Demo page, one child side by side | So what goes wrong? The agents keep testing when they should be teaching: two thirds of their turns are quizzes, right up to the exam. They move on after a lucky guess. And the damage is done on Wednesday, in the middle of the week. |
| 1:40 | Slide 10, sensitivity | We checked this is not an artefact of our numbers: across 27 versions of the simulator, the agents’ habit loses every time. They don’t adapt. |
| 1:52 | Slide 12, one command | Everything reruns with one command through Harbor. The Hidden-State Classroom: Sauna Renaissance. |

Recording tips: QuickTime → New Screen Recording, or Cmd+Shift+5. Read the script once aloud with a timer
before recording; cut words, not speed, if it runs over 2:00.

## 10-minute live demo

| min | what | where |
|---|---|---|
| 0–1 | The question and the punchline | slides 1–2 |
| 1–3 | What is hidden, why it is fair, how it is scored | slides 3–4 |
| 3–4 | The result and the disclosure test | slides 5–6 |
| 4–7 | **Live:** open the demo page, pick seed 3002, child 1. Play PPO and one agent side by side; point at the red marks (lucky-guess move-ons) and the quiz-heavy last turns; show the hidden state the agent never saw | `viz/index.html` |
| 7–8 | The three mistakes in numbers | slides 7–9 |
| 8–9 | Robustness and honest limits | slides 10–11 |
| 9–10 | One command; answer questions | slide 12, terminal |

For the judges’ rerun: `./eval.sh <model> 3001 3002 3003` (see README). A full three-seed run takes about an hour
with a Gemini model; have the seed-3002 result table ready on screen.

## Questions to expect

- **Is the reference fair?** It is a PPO policy that sees only the agent’s observations, through the same MCP tools.
  It was trained on seeds outside the evaluation range, and evaluation children are drawn from salted seeds.
- **Why not just give the agent more information?** We did: every parameter. No improvement.
- **Is the simulator realistic?** The mechanisms are documented effects; some numbers are our settings. The
  sensitivity analysis shows the agents lose in all 27 variations. It is not fitted to real student data (next step).
- **Can the agent game the reward?** Do-nothing and stop-early score 0. The state lives in a separate container, a
  separate verifier scores it, and two exploits found in review are fixed.
- **Could a smaller model learn this?** The PPO reference is itself a tiny network trained only on the environment’s
  experience; it beats every frontier model.
