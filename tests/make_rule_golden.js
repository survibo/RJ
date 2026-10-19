// rule-lab.html 의 RuleCore 로 src/rules.py 검증용 정답을 만든다.
//   node tests/make_rule_golden.js > tests/fixtures/rule_lab_golden.json
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'rule-lab.html'), 'utf8');
const core = html.match(/<script id="rule-core">([\s\S]*?)<\/script>/)[1];
const ctx = {};
vm.createContext(ctx);
vm.runInContext(core, ctx);
const { RULES, FUNCTIONS, runPipeline, rulesSpec } = ctx.RuleCore;

// 재현 가능한 난수 (mulberry32)
let seed = 12345;
function rand() {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
const pick = xs => xs[int(0, xs.length - 1)];

function randomStep(code) {
  const def = RULES.find(r => r.code === code);
  const fn = pick(FUNCTIONS).key;
  return {
    code,
    mode: pick(def.options).key,
    c: String(int(1, 5)),
    k: String(code === 'B' ? int(0, 9) : code === 'Q' ? int(1, 8) : int(1, 4)),
    a: String(int(1, 5)),
    b: String(int(1, 5)),
    t: String(int(1, 60)),
    period: Array.from({ length: int(1, 3) }, () => int(1, 5)).join(', '),
    fn,
    fc: String(fn === 'pow' ? int(1, 3) : int(1, 5)),
  };
}

function randomInput() {
  const n = int(3, 60);
  const m = int(1, Math.min(8, n));
  const pool = Array.from({ length: n }, (_, i) => i + 1);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = int(0, i);
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return { n, values: pool.slice(0, m) };
}

const codes = RULES.map(r => r.code);
const chains = [];
for (const code of codes) for (let i = 0; i < 30; i++) chains.push([code]);
for (let i = 0; i < 600; i++) chains.push(Array.from({ length: int(2, 5) }, () => pick(codes)));

const cases = chains.map(chain => {
  const { n, values } = randomInput();
  const steps = chain.map(randomStep);
  const result = runPipeline({ raw: values.join(', '), n: String(n), steps });
  const out = { n, values, steps, spec: rulesSpec(steps) };
  if (result.error) out.error = true;
  else out.final = result.final.map(Number);
  return out;
});

process.stdout.write(JSON.stringify(cases) + '\n');
