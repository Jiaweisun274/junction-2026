// JavaScript port of probe/tutor2.py (one child). Uses the child's exported random draws, so given the same
// actions it reproduces the Python simulator exactly. Each action also returns an "inside" record that explains
// what happened in the child's head (for the game's x-ray view).
(function (root) {
  function makeSim(D, c) {
    const C = D.const, K = D.concepts.length, TP = C.TURNS_PER, T = TP * C.SESSIONS.length;
    const BUGS = D.bugs, CARRY = D.carry;
    const s = {
      K, T, TP, turn: 0, state: Array(K).fill("unlearned"), h: Array(K).fill(0), last: Array(K).fill(0),
      lastExplained: null, sessExpl: {},
    };
    for (let k = 0; k < c.m0; k++) {
      const b = BUGS[k];
      if (b && c.suscept[b] && c.u_init[k] < 0.6) s.state[k] = "buggy";
      else { s.state[k] = "mastered"; s.h[k] = c.h_init[k]; s.last[k] = -3.0; }
    }
    s.now = function (turn) {
      const t = turn === undefined ? s.turn : turn;
      const i = Math.min(Math.floor(t / TP), C.SESSIONS.length - 1);
      return C.SESSIONS[i] + 0.01 * (t % TP);
    };
    s.recall = function (k, at) {
      if (s.state[k] !== "mastered") return 0;
      const a = at === undefined ? s.now() : at;
      return Math.pow(2, -(a - s.last[k]) / s.h[k]);
    };
    s.prereqOk = (k) => k === 0 || s.state[k - 1] === "mastered";
    function learn(k, uBug) {
      const b = BUGS[k];
      if (b && c.suscept[b] && uBug < 0.6) { s.state[k] = "buggy"; return "buggy"; }
      s.state[k] = "mastered"; s.h[k] = c.h0[k]; s.last[k] = s.now(); return "mastered";
    }
    s.explain = function (k) {
      const u = c.u[s.turn], st = s.state[k];
      const key = Math.floor(s.turn / TP) + ":" + k;
      const n = s.sessExpl[key] || 0;
      const damp = Math.pow(C.MASSED, n);
      s.sessExpl[key] = n + 1;
      const info = { act: "explain", k, before: st, nThisSession: n + 1, damp };
      if (st === "unlearned") {
        const base = s.prereqOk(k) ? c.plearn[k] : C.LEAK;
        info.prereqOk = s.prereqOk(k); info.chance = base * damp;
        if (u[0] < info.chance) { info.result = learn(k, u[1]); info.mech = info.result === "buggy" ? "misconception" : (n ? "massed" : "learn"); }
        else { info.result = "no"; info.mech = !info.prereqOk ? "prereq" : (n ? "massed" : "learn"); }
      } else if (st === "mastered") {
        const r = s.recall(k), sp = 1 + (1 - r);
        info.hBefore = s.h[k]; s.h[k] *= C.RESTUDY * sp; s.last[k] = s.now(); info.hAfter = s.h[k];
        info.result = "restudy"; info.mech = "forgetting";
      } else {
        info.chance = C.FIX_EXPLAIN;
        if (u[0] < C.FIX_EXPLAIN) { s.state[k] = "mastered"; s.h[k] = c.h0[k]; s.last[k] = s.now(); info.result = "fixed"; }
        else info.result = "still_buggy";
        info.mech = "misconception";
      }
      s.turn += 1; s.lastExplained = k;
      return info;
    };
    s.quiz = function (k) {
      const u = c.u[s.turn], st = s.state[k];
      const [q, right, bugAns, wrong] = D.quiz[k];
      const info = { act: "quiz", k, before: st };
      let ok;
      if (st === "mastered") {
        const rp = s.recall(k); info.recall = rp;
        if (u[0] < rp) {
          ok = u[1] < 1 - C.SLIP;
          if (ok) { info.hBefore = s.h[k]; s.h[k] *= C.RETRIEVE * (1 + (1 - rp)); info.hAfter = s.h[k]; info.why = "recalled"; info.mech = "testing"; }
          else { info.why = "slip"; info.mech = "noisy"; }
          s.last[k] = s.now();
        } else {
          ok = u[1] < C.GUESS; s.last[k] = s.now(); info.why = ok ? "forgot_guess" : "forgot"; info.mech = "forgetting";
        }
      } else if (st === "buggy") { ok = u[1] < 0.10; info.why = "buggy"; info.mech = "misconception"; }
      else { ok = u[1] < C.GUESS; info.why = ok ? "lucky" : "unlearned"; info.mech = "noisy"; }
      const carried = (k in CARRY) && s.state[CARRY[k]] === "buggy" && st !== "mastered";
      if (carried) { ok = u[1] < 0.10; info.why = "carried"; info.mech = "misconception"; }
      let ans;
      if (ok) ans = right;
      else if (st === "buggy" || carried) ans = bugAns;
      else ans = wrong[Math.floor(u[2] * wrong.length)];
      info.ok = ok; info.ans = ans; info.right = right; info.q = q; info.sig = !ok && ans === bugAns && bugAns !== null;
      s.turn += 1; s.lastExplained = null;
      return info;
    };
    s.correct = function (k, bug) {
      const u = c.u[s.turn], st = s.state[k];
      const info = { act: "correct", k, bug, before: st, mech: "misconception" };
      if (st === "buggy" && BUGS[k] === bug && u[0] < C.FIX_TARGETED) {
        s.state[k] = "mastered"; s.h[k] = c.h0[k]; s.last[k] = s.now(); info.result = "fixed";
      } else info.result = st === "buggy" && BUGS[k] === bug ? "not_this_time" : "nothing_to_fix";
      s.turn += 1; s.lastExplained = null;
      return info;
    };
    s.ask = function (k) {
      const u = c.u_ask[s.turn][k], st = s.state[k];
      let py;
      if (st === "buggy") py = 0.97;
      else if (st === "mastered" && u[0] < s.recall(k)) py = 0.95;
      else py = s.lastExplained === k ? 0.90 : 0.75;
      const yes = u[1] < py, tmpl = yes ? D.yes : D.no;
      return { act: "ask", k, before: st, yes, text: tmpl[Math.floor(u[2] * tmpl.length)], mech: "selfreport" };
    };
    s.examK = function (k) {
      const st = s.state[k];
      if (st === "mastered") { const p = s.recall(k, C.EXAM_DAY); return 0.9 * p + 0.25 * (1 - p); }
      return st === "buggy" ? C.BUG_EXAM : 0.25;
    };
    s.exam = function () { let x = 0; for (let k = 0; k < K; k++) x += s.examK(k); return x / K; };
    s.apply = function (a) {
      if (a[0] === "explain") return s.explain(a[1]);
      if (a[0] === "quiz") return s.quiz(a[1]);
      return s.correct(a[1], a[2]);
    };
    return s;
  }
  root.makeSim = makeSim;
  if (typeof module !== "undefined") module.exports = { makeSim };
})(typeof window !== "undefined" ? window : globalThis);
