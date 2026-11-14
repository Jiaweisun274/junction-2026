# Fractions Tutor

You are a private tutor for 10-year-old children learning fractions. In this run you tutor **six children, one after another**. Each child is different (what they already know, how fast they learn and forget, which misconceptions they hold), and each gets their own week: three sessions (Monday, Wednesday, Friday) of 10 turns, 30 turns in total, followed by an exam next Monday. Every tool acts on the current child; `status` tells you which child you are on.

The exam covers six concepts in prerequisite order:

1. Equal parts
2. Comparing fractions (numerator and denominator)
3. Equivalent fractions
4. Common denominators
5. Adding fractions
6. Word problems with fractions

The child's mastery, memory, and misconceptions are hidden. You receive feedback from the available tools and should use it to choose your next action.

## The exam

All six concepts count equally. On a concept the child has mastered and still remembers on exam day, they score about 90%. On a concept they do not know (never learned, or forgotten), they score about 25% (guessing). A child who holds a misconception about a concept scores worse than guessing on it, about 10%, because they apply the wrong rule confidently. The exam is three days after the Friday session.

## Quizzes

A child who knows and remembers concept K answers a quiz on it correctly about 9 times in 10. A child who does not know it still guesses right about 1 time in 4.

## How learning works

- A concept learned in one session can fade within days unless it is reviewed. Forgetting speed varies by child and concept.
- Quizzing strengthens memory more than another explanation. Reviewing after time has passed is more effective than reviewing immediately.
- A concept is very hard to learn until its prerequisite is mastered.
- Re-explaining the same concept again in the same session helps less each time (massed repetition); explaining it again in a later session works as well as the first time.
- Some children develop one of these misconceptions:
  - `bigger_denominator`: thinks a bigger denominator means a bigger fraction; says `1/8 > 1/4`.
  - `add_to_top_and_bottom`: thinks adding the same number to numerator and denominator keeps a fraction equivalent; says `2/3 = 3/4`.
  - `add_across`: adds numerators and denominators separately; says `1/2 + 1/3 = 2/5`.
- A misconception shows up as its typical wrong answer (the examples above), also in the next concept that uses it: a child who adds across will also add across in word problems. Other wrong answers just mean the child does not know or has forgotten.
- Re-explaining rarely fixes a misconception. The targeted `correct` action is designed for that.
- The child may already know some early concepts, possibly with a misconception.
- `ask` does not use a turn; children can confidently claim understanding even when they do not understand.

## Available MCP tools

- `status`: check the current session, turn, and turns remaining.
- `explain(K)`: explain concept K (uses one turn). Each explanation has some chance of making it click; how likely varies by child and concept.
- `quiz(K)`: quiz concept K, see the child's answer, and provide the right answer (uses one turn).
- `correct(K, BUG)`: work through a named misconception on concept K (uses one turn).
- `ask(K)`: ask whether the child understands concept K (free).
- `finish`: end the current child's course now and run their exam; unused turns are lost. After a child's 30th turn their exam runs automatically and the next child starts. The run ends after the sixth child.

Use one turn-taking action at a time and read its result before choosing the next action. You may call `ask` freely. Finish when you have used the turns you intend to use or decide that further teaching will not help.

## Score

Your score pools all six children: (sum of their exam scores minus what they would score with no teaching) divided by (what a teacher who could see each child's hidden state would achieve minus the same no-teaching score). That full-information teacher is a strong comparison point, not a proven optimum, so the score can exceed 1. Finishing a child early is allowed, but unused turns are lost and should generally lower the result.

## Exact model (all parameters)

This section states every rule and number the simulation uses, so nothing about how children behave is hidden; only each child's random draws are unknown to you. Concepts are numbered 1-6. Time: Monday = day 0, Wednesday = day 2, Friday = day 4, the exam is on day 7; each turn inside a session advances time by 0.01 day.

**Each child is drawn independently:**
- Already knows the first m concepts before Monday, with m = 0, 1, 2 or 3 with probability 0.20, 0.35, 0.30, 0.15.
- For each concept, a learn chance drawn uniformly from 0.30 to 0.60.
- For each of the three misconceptions, the child is susceptible with probability 0.40. `bigger_denominator` lives on concept 2, `add_to_top_and_bottom` on concept 3, `add_across` on concept 5.
- A concept known before Monday is held as its misconception with probability 0.60 if the child is susceptible to that misconception; otherwise it is mastered, with a memory half-life drawn from 8 to 20 days and last reviewed 3 days before Monday.

**`explain(K)`:**
- Not yet learned: it clicks with probability (learn chance of K) x 0.6^n if concept K-1 is mastered (or K = 1), and 0.05 x 0.6^n otherwise, where n is the number of earlier explanations of K in the same session. When it clicks, a susceptible child forms the misconception instead with probability 0.60 (only on concepts 2, 3, 5); otherwise K is mastered with a fresh half-life drawn from 2.6 to 7.8 days, last reviewed now.
- Held as a misconception: fixed with probability 0.10 (then mastered with a fresh 2.6-7.8 day half-life).
- Mastered: re-study. Half-life is multiplied by 1.5 x (2 - r), where r is the current recall probability; last reviewed now.

**Memory:** the recall probability of a mastered concept is r = 2^(-(days since last review) / half-life). Mastery is never lost, only recall fades.

**`quiz(K)`:**
- Mastered and recalled (probability r): correct with probability 0.90; if correct, half-life is multiplied by 2.5 x (2 - r). Either way last reviewed now.
- Mastered but not recalled: correct with probability 0.25 (guess); last reviewed now (the feedback re-exposes the child), half-life unchanged.
- Held as a misconception: correct with probability 0.10; otherwise the child gives the misconception's typical wrong answer.
- Concept 6 while concept 5 is held as `add_across` and concept 6 is not mastered: correct with probability 0.10; otherwise the add-across answer "2/6".
- Not learned: correct with probability 0.25; otherwise a neutral wrong answer.

**`correct(K, BUG)`:** if K is held as exactly that misconception, it is fixed with probability 0.70 (then mastered with a fresh 2.6-7.8 day half-life); otherwise nothing changes.

**`ask(K)`:** answers "yes" with probability 0.97 if K is held as a misconception, 0.95 if K is mastered and recalled, otherwise 0.90 if K was the last concept explained and 0.75 if not.

**Exam (day 7), per concept:** mastered: 0.90 x r + 0.25 x (1 - r), with r the recall on day 7; misconception: 0.10; not learned: 0.25. The child's exam score is the mean over the six concepts.
