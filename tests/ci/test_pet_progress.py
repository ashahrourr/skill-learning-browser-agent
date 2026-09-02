"""Real-browser tests for the pet's self-defined numeric progress engine.

The pet picks its OWN success signal per task: a tiny JS expression that reads a single number
off the live page (higher = closer to done) plus the goal value. These tests prove that engine on
a NON-game page — a to-do list whose task is "check off every item" — with a real browser and no
mocks (only the LLM that would author the metric is skipped; here we supply the metric directly,
which is exactly what _create_progress_plan would emit).
"""

import asyncio

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile, ViewportSize
from browser_use.pet import AuthoredHelper, PetBridge, PetSession, _format_metric, _HelperContext

# Four to-do items, none checked. The pet's number = how many boxes are checked; goal = 4 (all done).
TODO_HTML = """
<html><body>
	<h1>My To-Do List</h1>
	<ul id="todos">
		<li><input type="checkbox" class="item"> Buy milk</li>
		<li><input type="checkbox" class="item"> Walk dog</li>
		<li><input type="checkbox" class="item"> Write tests</li>
		<li><input type="checkbox" class="item"> Ship it</li>
	</ul>
</body></html>
"""

# The exact kind of metric the LLM is asked to write: a pure, side-effect-free expression returning
# a number by reading the DOM. Counts checked items.
CHECKED_COUNT_JS = "document.querySelectorAll('.item:checked').length"
GOAL = 4.0


# A typing surface that ONLY records real keystrokes: it counts keydown events and appends each typed
# character itself (it never trusts input.value), exactly like Monkeytype counts real keypresses. Setting
# .value would move neither counter — proving ctx.type_text fires genuine keys, not a synthetic value-set.
TYPING_HTML = """
<html><body>
	<input id="field" autofocus>
	<script>
		window.__keydowns = 0;
		window.__typed = "";
		const f = document.getElementById('field');
		f.focus();
		f.addEventListener('keydown', (e) => {
			window.__keydowns += 1;
			if (e.key && e.key.length === 1) window.__typed += e.key;
		});
	</script>
</body></html>
"""


@pytest.fixture(scope='module')
def todo_server():
	server = HTTPServer()
	server.start()
	server.expect_request('/todos').respond_with_data(TODO_HTML, content_type='text/html')
	server.expect_request('/typing').respond_with_data(TYPING_HTML, content_type='text/html')
	yield server
	server.stop()


@pytest.fixture
async def browser_session():
	session = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=True,
			window_size=ViewportSize(width=800, height=600),
		)
	)
	await session.start()
	yield session
	await session.kill()


@pytest.fixture
def bridge(tmp_path):
	return PetBridge(cdp_url=None, max_steps=10, log_dir=tmp_path / 'logs', site_dir=tmp_path / 'sites')


async def _check_items(browser_session: BrowserSession, count: int) -> None:
	"""Tick the first `count` checkboxes — stands in for the agent acting on the page."""
	cdp_session = await browser_session.get_or_create_cdp_session(focus=True)
	await cdp_session.cdp_client.send.Runtime.evaluate(
		params={
			'expression': f"document.querySelectorAll('.item').forEach((b, i) => {{ b.checked = i < {count}; }})",
			'returnByValue': True,
		},
		session_id=cdp_session.session_id,
	)


async def test_progress_reads_live_number_and_climbs(bridge, browser_session, todo_server):
	"""The metric reads the real count off the page, and rises as the page changes — no screenshot."""
	await browser_session.navigate_to(f'http://{todo_server.host}:{todo_server.port}/todos')

	# Nothing checked yet -> score 0, well below goal.
	score = await bridge._read_progress(browser_session, CHECKED_COUNT_JS)
	assert score == 0.0
	assert score < GOAL  # not done

	# Agent checks two items -> the SAME metric now reads 2, straight from the DOM.
	await _check_items(browser_session, 2)
	assert await bridge._read_progress(browser_session, CHECKED_COUNT_JS) == 2.0

	# Agent finishes the list -> score reaches the goal, which is how the loop knows it's done.
	await _check_items(browser_session, 4)
	final = await bridge._read_progress(browser_session, CHECKED_COUNT_JS)
	assert final == GOAL
	assert final >= GOAL  # done


async def test_bad_metric_fails_soft_to_none(bridge, browser_session, todo_server):
	"""A metric that throws or returns a non-number yields None, so the loop falls back to vision."""
	await browser_session.navigate_to(f'http://{todo_server.host}:{todo_server.port}/todos')

	assert await bridge._read_progress(browser_session, 'this.is.not.valid.js(') is None  # throws
	assert await bridge._read_progress(browser_session, "'a string, not a number'") is None
	assert await bridge._read_progress(browser_session, 'true') is None  # bool is not progress


async def test_page_snapshot_feeds_metric_design(bridge, browser_session, todo_server):
	"""The snapshot handed to the metric-designer LLM contains the real page text it must reference."""
	await browser_session.navigate_to(f'http://{todo_server.host}:{todo_server.port}/todos')
	text = await bridge._page_text_snapshot(browser_session)
	assert 'My To-Do List' in text
	assert 'Buy milk' in text


def test_skill_save_routing_matches_run_shape(bridge):
	"""How a run is codified is decided by the run's SHAPE — read off the agent's own tool use, not
	the site. This is the fix for the monkeytype bug: a code-shaped run that never produced a working
	helper must NOT fall through to the trace re-derivation (which saved a broken typing skill)."""
	session = PetSession(session_id='s1')

	# Pure click/type workflow: the agent never reached for code -> trace re-derivation fits.
	assert bridge._skill_save_decision(session) == 'trace'

	# Code-shaped but unproven: the agent tried helpers and they FAILED (recorded in attempt_log, not as
	# authored_helpers). Saving a trace-derived click skill here is the wrong shape -> save nothing.
	session.attempt_log.append('1. read words and type them -> runtime error: ...')
	assert bridge._skill_save_decision(session) == 'none'

	# A helper actually worked: save that verified code regardless of earlier failed attempts.
	session.authored_helpers.append(AuthoredHelper(code='async def run(ctx):\n\treturn None', element_indices=[]))
	assert bridge._skill_save_decision(session) == 'helper'


async def test_type_text_fires_real_keystrokes(browser_session, todo_server):
	"""ctx.type_text types real per-key events, so a site that counts keydowns (Monkeytype, games) sees
	them — the exact thing the old saved skill's .value-set typing failed to do, which left it stuck."""
	await browser_session.navigate_to(f'http://{todo_server.host}:{todo_server.port}/typing')

	# Focus the typing surface, then type a string through the helper's real-keystroke path.
	cdp = await browser_session.get_or_create_cdp_session(focus=True)
	await cdp.cdp_client.send.Runtime.evaluate(
		params={'expression': "document.getElementById('field').focus()", 'returnByValue': True},
		session_id=cdp.session_id,
	)
	ctx = _HelperContext(browser_session, [], asyncio.Event())
	await ctx.type_text('hello world')

	state = await ctx.js('({kd: window.__keydowns, typed: window.__typed, value: document.getElementById("field").value})')
	# One real keydown per character (11 incl. the space) — proof these were genuine keypresses.
	assert state['kd'] == len('hello world')
	# The page reconstructed the text purely from keydown events it observed (never from .value).
	assert state['typed'] == 'hello world'
	# And the characters actually landed in the field.
	assert state['value'] == 'hello world'


async def test_autopilot_runs_in_page_and_reports_death(bridge, browser_session, todo_server):
	"""The reflex loop runs INSIDE the page on its own clock (setInterval), updates the result global
	every tick, ends itself, and reports back its score + DEATH state — all with no per-move round-trip.
	This is the fast-loop half of real-time play; Python only injects + polls + reads the death."""
	await browser_session.navigate_to(f'http://{todo_server.host}:{todo_server.port}/todos')
	session = PetSession(session_id='ap1')

	# A tiny self-running "game": score ticks up each frame; at 5 it "dies" with a death object.
	setup_js = """
	let score = 0;
	const id = setInterval(() => {
		const a = window.__PET_AUTOPILOT;
		if (window.__PET_AUTOPILOT_STOP) { clearInterval(id); return; }
		score += 1;
		a.score = score;
		a.ticks = (a.ticks || 0) + 1;
		if (score >= 5) { a.death = {reason: 'reached_cap', at: score}; a.done = true; clearInterval(id); }
	}, 20);
	"""
	result = await bridge._run_autopilot(session, setup_js, max_seconds=10, browser_session=browser_session)

	assert result.error is None, result.error
	assert 'score=5' in result.extracted_content  # the in-page loop drove the score, read back over CDP
	assert 'reached_cap' in result.extracted_content  # the death state — the lesson the LLM rewrites against
	assert len(session.attempt_log) == 1  # recorded like a helper attempt so the LLM remembers it


async def test_autopilot_times_out_and_stops_the_loop(bridge, browser_session, todo_server):
	"""A loop that never reports done is cut off at max_seconds, told to stop, and reported as timed out —
	so a runaway reflex can't keep playing into the next step."""
	await browser_session.navigate_to(f'http://{todo_server.host}:{todo_server.port}/todos')
	session = PetSession(session_id='ap2')

	# Never sets done — just keeps counting. Must honor __PET_AUTOPILOT_STOP so it halts when we cut it off.
	setup_js = """
	const id = setInterval(() => {
		const a = window.__PET_AUTOPILOT;
		if (window.__PET_AUTOPILOT_STOP) { clearInterval(id); return; }
		a.score = (a.score || 0) + 1;
		a.ticks = (a.ticks || 0) + 1;
	}, 20);
	"""
	result = await bridge._run_autopilot(session, setup_js, max_seconds=1, browser_session=browser_session)
	assert 'timed out' in result.extracted_content

	# After the stop signal the loop must be halted — score stops advancing.
	cdp = await browser_session.get_or_create_cdp_session(focus=True)
	first = await bridge._read_progress(browser_session, 'window.__PET_AUTOPILOT.score')
	await cdp.cdp_client.send.Runtime.evaluate(params={'expression': '0', 'returnByValue': True}, session_id=cdp.session_id)
	second = await bridge._read_progress(browser_session, 'window.__PET_AUTOPILOT.score')
	assert first == second  # frozen — the loop is no longer running


def test_format_metric_prints_cleanly():
	"""Whole numbers print without a trailing .0; real fractions keep precision."""
	assert _format_metric(512.0) == '512'
	assert _format_metric(4.0) == '4'
	assert _format_metric(0.873) == '0.873'
	assert _format_metric(float('-inf')) == '-'
