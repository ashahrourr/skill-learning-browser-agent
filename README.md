# Website Pet

**A browser agent that learns a site once, then replays it — with a cat that walks to every element
before it touches it.**

![The pet filling in a job application form field by field and submitting it](docs/pet/cat-demo.gif)

---

## The problem

A browser agent re-solves the same page from scratch on every run. The same LLM calls, the same
cost, the same latency — and no guarantee it does the same thing twice, because nothing it learned
last time survived.

It also works invisibly. The tab twitches, things get clicked, and you reconstruct what happened
afterwards from a log. You cannot supervise it, and you cannot stop it in time.

## The solution

**Learn the site once.** Runs are recorded as traces. Repeated traces become a validated, reusable
skill for that site, so the second visit is a replay instead of a fresh exploration.

**Show the work.** A pixel cat walks to each element before the agent acts on it, and a panel
narrates every step. You watch it work, and there is a Stop button under your cursor the whole time.

## How it works

**A skill is a schema, not generated code.** The obvious approach — have the LLM write a script —
fails in repetitive ways: invented selector syntax, hardcoded month names, brittle date parsing.
`pet_skill_steps.py` makes those unrepresentable. A skill is a validated list of steps run by one
interpreter, so the model fills in a structure it cannot violate.

**Learning is proposal-only.** `pet_skill_learner.py` never installs what it learns — it writes a
candidate for review (`--propose-skill example.com`). An agent that silently rewrites its own
behaviour after a run it *thinks* went well is one you cannot trust.

**It never drives the page by accident.** `key-trap.js` runs at `document_start` and swallows
keystrokes originating in the pet UI. Without it, typing a task while sitting on a game or an editor
would drive the page underneath — arrow keys moving tiles while you type.

## What I built

| Path | Lines | |
|---|---:|---|
| `browser_use/pet.py` | 2,359 | bridge on `127.0.0.1:8765` — task lifecycle, questions, stop |
| `browser_use/pet_skill_learner.py` | 453 | turns traces into proposed skills |
| `browser_use/pet_skill_context.py` | 423 | the surface a skill is allowed to touch |
| `browser_use/pet_trace.py` | 423 | run traces — the training data for skills |
| `browser_use/pet_skills.py` | 280 | skill registry: load, match, execute |
| `browser_use/pet_skill_steps.py` | 277 | step schema + interpreter |
| `browser_use/pet_memory.py` | 249 | per-site memory and reflections |
| `browser_use/pet_extension/` | — | Chrome extension: PixiJS cat, task panel, key trap |
| `browser_use/llm/openclaw/` | — | `ChatOpenClaw` provider |
| `tests/ci/test_pet_*.py` | 1,232 | tests |

Plus the browser-use internals the pet needs: coordinate clicking, DOM serializer output, watchdog
and tools changes.

## Run it

Python 3.11+, and a Chrome started with remote debugging.

```bash
uv sync                    # or: pip install -e .
python -m browser_use.pet  # bridge on 127.0.0.1:8765
```

Load `browser_use/pet_extension/` at `chrome://extensions` → Developer mode → **Load unpacked**.
Open any page, click the extension icon to deploy the pet there, then click the cat and type a task.

Running real tasks needs an LLM configured for `ChatOpenClaw`. The demo above was recorded with
scripted model replies, so the recording shows the pet layer rather than a live agent.

<sub>Built on [browser-use](https://github.com/browser-use/browser-use) (MIT).</sub>
