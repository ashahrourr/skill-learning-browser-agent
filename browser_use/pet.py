"""Serve tasks from the Website Pet Chrome extension to browser-use."""

import argparse
import asyncio
import base64
import contextvars
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from aiohttp import web
from aiohttp.abc import AbstractAccessLogger
from pydantic import BaseModel, Field

from browser_use import ActionResult, Agent, Browser, BrowserProfile, BrowserSession, ChatOpenClaw, Tools
from browser_use.browser.events import (
	ClickElementEvent,
	ScrollEvent,
	SendKeysEvent,
	TypeTextEvent,
)
from browser_use.dom.views import EnhancedDOMTreeNode
from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage
from browser_use.pet_memory import PetSiteMemory, SiteMemoryReflection
from browser_use.pet_skill_learner import PetSkillLearner
from browser_use.pet_skills import PetSkillRegistry
from browser_use.pet_trace import PetTraceBuilder
from browser_use.skill_cli.utils import discover_chrome_cdp_url


def _load_pet_prompt(file_name: str) -> str:
	"""Load a pet prompt from package data."""
	return resources.files('browser_use.pet_prompts').joinpath(file_name).read_text(encoding='utf-8').strip()


_PET_DEFAULT_MODEL = 'openai-codex/gpt-5.5'

# Steps-based skill generation (re-deriving a skill from trace text) is disabled in favor of
# helpers the agent authors and verifies live. Flip to True to re-enable the old path for comparison.
_STEPS_SKILL_GENERATION_ENABLED = False


def _pet_llm() -> ChatOpenClaw:
	"""Build the pet LLM, overridable via the PET_MODEL env var (provider/model id from the openclaw catalog)."""
	return ChatOpenClaw(model=os.environ.get('PET_MODEL', _PET_DEFAULT_MODEL))


_PET_STEP_TIMEOUT_SECONDS = 24 * 60 * 60
# openclaw subscription models can be slow; the default 75s LLM-call cap kills steps mid-thought.
# Give each LLM call more room so a slow-but-working model completes instead of timing out.
_PET_LLM_TIMEOUT_SECONDS = 300
_MAX_SEQUENTIAL_UNITS = 20
# Hard safety cap on the verify→fix→re-verify loop. The loop normally stops earlier — either it passes
# or the match score stops improving (stuck). This just bounds the worst case so it can't spin forever.
_MAX_FIX_ATTEMPTS = 8
_SAFETY_INSTRUCTIONS = '\n\n' + _load_pet_prompt('execution_policy.md')
_TASK_PARSER_SYSTEM_PROMPT = _load_pet_prompt('task_parser.md')
_QUICK_APPLY_INSTRUCTION = _load_pet_prompt('quick_apply.md')
_PET_LOG_SESSION: contextvars.ContextVar[str | None] = contextvars.ContextVar('pet_log_session', default=None)


def _safe_log_name(session_id: str) -> str:
	"""Convert a session id into a stable local log filename."""
	return re.sub(r'[^a-zA-Z0-9_.-]+', '-', session_id).strip('-') or 'unknown-session'


class PetSessionLogHandler(logging.Handler):
	"""Route detailed browser-use logs into the current pet session's log file."""

	def __init__(self, log_dir: Path):
		super().__init__(logging.INFO)
		self.log_dir = log_dir
		self.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s [%(name)s] %(message)s'))

	def emit(self, record: logging.LogRecord) -> None:
		session_id = _PET_LOG_SESSION.get()
		if not session_id:
			return
		try:
			self.log_dir.mkdir(parents=True, exist_ok=True)
			log_path = self.log_dir / f'{_safe_log_name(session_id)}.log'
			with log_path.open('a', encoding='utf-8') as log_file:
				log_file.write(self.format(record) + '\n')
		except Exception:
			self.handleError(record)


class PetTask(BaseModel):
	"""Task submitted from a visible browser tab."""

	id: str = Field(default_factory=lambda: str(uuid4()))
	session_id: str = Field(min_length=1, max_length=200)
	task: str = Field(min_length=1, max_length=4000)
	url: str = Field(min_length=1, max_length=8000)
	origin_token: str | None = Field(default=None, min_length=1, max_length=200)


class PetReply(BaseModel):
	"""Answer submitted by the user for an active pet question."""

	session_id: str = Field(min_length=1, max_length=200)
	reply: str = Field(min_length=1, max_length=4000)


class PetStatus(BaseModel):
	"""Visible state of the local pet agent."""

	state: Literal['idle', 'queued', 'running', 'waiting', 'stopping', 'completed', 'stopped', 'failed'] = 'idle'
	message: str = 'Ready'
	question: str | None = None
	task_id: str | None = None


@dataclass
class PetSession:
	"""Per-tab pet state managed by the shared bridge process."""

	session_id: str
	queue: asyncio.Queue[PetTask] = field(default_factory=asyncio.Queue)
	status: PetStatus = field(default_factory=PetStatus)
	reply_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
	cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
	current_agent: Agent | None = None
	cancelled_task_ids: set[str] = field(default_factory=set)
	runner_task: asyncio.Task[None] | None = None
	visible_steps: list[str] = field(default_factory=list)
	authored_helpers: list['AuthoredHelper'] = field(default_factory=list)
	# Compact "what I tried this run and how it went" — re-shown on each helper result so it
	# survives context-trimming and the agent stops repeating failed approaches.
	attempt_log: list[str] = field(default_factory=list)
	# The agent's live notepad for THIS run: short strategy lessons it writes via the `note` action
	# and re-reads each step, so it learns and adapts WITHIN the run (e.g. "swiping up breaks the
	# corner — avoid it"). Distilled into long-term site memory at the end of a run.
	notes: list[str] = field(default_factory=list)


class ExecutionContract(BaseModel):
	"""Structured plan for running a submitted browser task."""

	mode: Literal['single', 'sequential'] = 'single'
	unit_task: str = Field(min_length=1, max_length=4000)
	total_units: int = Field(default=1, ge=1, le=_MAX_SEQUENTIAL_UNITS)
	approval_required: bool = False
	completion_condition: str = Field(min_length=1, max_length=1000)


class CompletionVerdict(BaseModel):
	"""Result of a one-shot visual check that a task was actually completed."""

	achieved: bool
	# 0-100 how close the result is to correct (100 = perfect). Used to tell whether retries are
	# making progress: a rising match means keep going, a stalled one means it's stuck — stop.
	match: int = Field(default=0, ge=0, le=100)
	reason: str = Field(default='', max_length=300)


class ProgressPlan(BaseModel):
	"""A self-defined, numeric success signal for the task — the pet's own scoreboard.

	Instead of us hardcoding what "done" means per site (image-match for Piskel, tile=512 for
	2048), the agent writes a tiny JS expression that reads its own progress off the page as a
	single number where HIGHER = closer to done, plus the value that counts as finished. The
	orchestrator re-reads that number for free every round (no LLM, no screenshot) to decide
	keep-going vs adapt-strategy vs done. If the task can't be reduced to a number (e.g. "make it
	look like this picture"), `usable=False` and we fall back to the visual judge — which then is
	just ONE kind of progress signal, not a special case baked into the loop.
	"""

	# True only if a meaningful numeric progress reader exists; False => use the visual judge instead.
	usable: bool
	# A JS expression that EVALUATES to a number (higher = more progress). Must be side-effect free and
	# read only what is visibly on the page, e.g. for 2048: the current max tile value.
	metric_js: str = Field(default='', max_length=2000)
	# The metric value at which the task is complete (e.g. 512). Reaching/exceeding it => done.
	goal: float = 0.0
	# One short human description of the metric, shown back to the agent ("max tile on the board").
	metric_label: str = Field(default='', max_length=200)
	# The first strategy to try, in the agent's own words ("keep the largest tile in a corner; cycle
	# left/down, only press up/right when forced"). Re-stated to the agent each round; rewritten on stall.
	strategy: str = Field(default='', max_length=1000)


class AuthoredHelper(BaseModel):
	"""A helper the agent wrote and ran live during a task.

	Element references are live browser_state indices the agent confirmed on the
	page, so the helper acts only on real elements — it cannot invent selectors.
	We keep each used element's fingerprint so the helper can be saved as a skill
	(indices change per page load; the fingerprint gives a stable selector).
	"""

	code: str
	element_indices: list[int]
	used_elements: list[dict[str, Any]] = Field(default_factory=list)
	# index -> stable selector, so a saved helper re-finds the same controls next run.
	index_selectors: dict[int, str] = Field(default_factory=dict)


def _format_metric(value: float) -> str:
	"""Print a metric value cleanly: 512 not 512.0, but keep real fractions (0.87)."""
	if value == float('-inf'):
		return '-'
	if float(value).is_integer():
		return str(int(value))
	return f'{value:.3g}'


def _stable_selector(fingerprint: dict[str, Any] | None) -> str | None:
	"""Best stable selector for a clicked element, from its recorded fingerprint.

	Prefers #id, then [aria-label="..."], then tag.firstclass, then the absolute x_path —
	whatever survives a page reload so a saved helper can re-find the control.
	"""
	if not fingerprint:
		return None
	if fingerprint.get('id'):
		return f'#{fingerprint["id"]}'
	if fingerprint.get('aria'):
		return f'[aria-label="{fingerprint["aria"]}"]'
	classes = fingerprint.get('classes') or []
	tag = fingerprint.get('tag') or ''
	if tag and classes:
		return f'{tag}.{classes[0]}'
	if fingerprint.get('x_path'):
		x_path = str(fingerprint['x_path'])
		return f'xpath:{x_path}'
	return None


class _HelperContext:
	"""Runtime passed to an agent-authored helper as `ctx`.

	Resolves a live browser_state index to its real CDP node and dispatches
	grounded clicks/typing through the same event path the agent uses. The helper
	may only touch indices the agent declared (and thus saw live), so a hallucinated
	index raises instead of silently acting on the wrong element.
	"""

	def __init__(self, browser_session: BrowserSession, allowed_indices: list[int], cancel_event: asyncio.Event):
		self._session = browser_session
		self._allowed = set(allowed_indices)
		self._cancel = cancel_event
		self.used_elements: list[dict[str, Any]] = []
		self.click_count = 0
		# index -> a stable selector for the element at that index, captured as the helper runs,
		# so a saved helper can re-find the same controls next run.
		self.index_selectors: dict[int, str] = {}
		self._trace_builder = PetTraceBuilder()

	async def _node(self, index: int) -> EnhancedDOMTreeNode:
		if self._cancel.is_set():
			raise RuntimeError('Task stopped by the user.')
		if index not in self._allowed:
			raise ValueError(f'index {index} was not listed in element_indices; only declared indices can be used')
		node = await self._session.get_element_by_index(index)
		if node is None:
			raise ValueError(f'index {index} is no longer on the page')
		fingerprint = self._trace_builder.element_fingerprint(node)
		if fingerprint and fingerprint not in self.used_elements:
			self.used_elements.append(fingerprint)
		# Remember a STABLE selector for this index so a saved helper can re-find the same
		# control next run (live indices change every page load; selectors/x_paths survive).
		if index not in self.index_selectors:
			selector = _stable_selector(fingerprint)
			if selector:
				self.index_selectors[index] = selector
		return node

	async def click(self, index: int, hold: float = 0.08) -> None:
		"""Click the element at a declared live index — fast, with a brief hold.

		Dispatches a raw CDP mouse press/release at the element's viewport center, with no
		post-click page/network-settle wait. This is what makes a helper fast enough for a
		song: the agent controls rhythm with ctx.wait, not the click overhead. (The agent's
		normal click goes through ClickElementEvent, which settles after every click — ~seconds
		each — far too slow for a timed sequence.)

		`hold` is how long the button stays down between press and release. A real key-press
		visual (the key lighting up via :active or a pressed class toggled on mousedown/mouseup)
		needs the button held for a perceptible moment, otherwise the highlighted state never
		paints. Defaults to 80ms; pass hold=0 for instant clicks where visuals don't matter, or
		a larger value for sustained notes.
		"""
		node = await self._node(index)
		cdp_session = await self._session.get_or_create_cdp_session(focus=True)
		session_id = cdp_session.session_id
		quads = await cdp_session.cdp_client.send.DOM.getContentQuads(
			params={'backendNodeId': node.backend_node_id},
			session_id=session_id,
		)
		quad = next(iter(quads.get('quads') or []), None)
		if not quad:
			raise ValueError(f'index {index} has no layout box to click')
		center_x = sum(quad[i] for i in range(0, 8, 2)) / 4
		center_y = sum(quad[i] for i in range(1, 8, 2)) / 4
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseMoved', 'x': center_x, 'y': center_y}, session_id=session_id
		)
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mousePressed', 'x': center_x, 'y': center_y, 'button': 'left', 'clickCount': 1},
			session_id=session_id,
		)
		await asyncio.sleep(max(0.0, min(float(hold), 2.0)))
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseReleased', 'x': center_x, 'y': center_y, 'button': 'left', 'clickCount': 1},
			session_id=session_id,
		)
		self.click_count += 1

	async def click_at(self, x: float, y: float, hold: float = 0.0) -> None:
		"""Click at a raw viewport coordinate with a REAL CDP mouse event.

		Use this to draw on a <canvas>: pixels are not elements with indices, so you draw the
		way a human does — point the (already-selected) pen at a screen coordinate and click,
		and the app paints whatever pixel is under it. These are real OS-level events, so the
		app accepts them (unlike synthetic ctx.js events, which many canvas apps ignore — that's
		why JS-painted pixels do not stick). Map your sprite pixel -> screen coordinate yourself
		from the canvas's on-screen box.
		"""
		cdp_session = await self._session.get_or_create_cdp_session(focus=True)
		session_id = cdp_session.session_id
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseMoved', 'x': x, 'y': y}, session_id=session_id
		)
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mousePressed', 'x': x, 'y': y, 'button': 'left', 'clickCount': 1}, session_id=session_id
		)
		await asyncio.sleep(max(0.0, min(float(hold), 2.0)))
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseReleased', 'x': x, 'y': y, 'button': 'left', 'clickCount': 1}, session_id=session_id
		)
		self.click_count += 1

	async def drag(self, path: list[tuple[float, float]], steps_between: int = 4) -> None:
		"""Drag the mouse through a path of viewport coordinates with the button held (a real stroke).

		For freehand pen strokes / lines on a canvas: presses at the first point, moves through
		the rest (interpolating intermediate moves so the stroke is continuous), releases at the
		last. Real CDP events, so the app's drawing tool registers it.
		"""
		if len(path) < 2:
			raise ValueError('drag needs at least 2 points')
		cdp_session = await self._session.get_or_create_cdp_session(focus=True)
		session_id = cdp_session.session_id
		start_x, start_y = path[0]
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseMoved', 'x': start_x, 'y': start_y}, session_id=session_id
		)
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mousePressed', 'x': start_x, 'y': start_y, 'button': 'left', 'clickCount': 1},
			session_id=session_id,
		)
		prev_x, prev_y = start_x, start_y
		for next_x, next_y in path[1:]:
			segments = max(1, steps_between)
			for step in range(1, segments + 1):
				interp_x = prev_x + (next_x - prev_x) * step / segments
				interp_y = prev_y + (next_y - prev_y) * step / segments
				await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
					params={'type': 'mouseMoved', 'x': interp_x, 'y': interp_y, 'button': 'left'}, session_id=session_id
				)
			prev_x, prev_y = next_x, next_y
		end_x, end_y = path[-1]
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseReleased', 'x': end_x, 'y': end_y, 'button': 'left', 'clickCount': 1},
			session_id=session_id,
		)
		self.click_count += 1

	async def type(self, index: int, text: str, clear: bool = True) -> None:
		"""Type text into the element at a declared live index."""
		node = await self._node(index)
		event = self._session.event_bus.dispatch(TypeTextEvent(node=node, text=text, clear=clear))
		await event
		await event.event_result(raise_if_any=True, raise_if_none=False)

	async def press(self, keys: str) -> None:
		"""Press a keyboard key or shortcut on the page (e.g. 'ArrowUp', 'Enter', 'ctrl+a').

		For key-driven UIs and games (arrow keys, WASD) — dispatched as a REAL key event so the
		page's keydown handlers fire (synthetic JS key events are often ignored). Pace a sequence
		of moves with ctx.wait between presses.
		"""
		if self._cancel.is_set():
			raise RuntimeError('Task stopped by the user.')
		# A focused text field — including the pet's own chat input — would EAT the key (arrow keys
		# move its cursor instead of the game). Blur the truly-focused element first (walking into any
		# shadow roots, since the pet overlay lives in one) so the press reaches the page's
		# document/window keydown handlers, which is where games like 2048 listen.
		await self.js(
			'(() => { let el = document.activeElement;'
			'  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;'
			'  if (el && el.blur) el.blur(); })()'
		)
		event = self._session.event_bus.dispatch(SendKeysEvent(keys=keys))
		await event
		await event.event_result(raise_if_any=True, raise_if_none=False)

	async def type_text(self, text: str, per_char_delay: float = 0.0) -> None:
		"""Type a string as REAL per-character keystrokes into whatever is focused on the page.

		Use this for sites that count ACTUAL keypresses — typing tests (Monkeytype), games, code
		editors — where setting an input's `.value` (synthetic) is silently ignored. Click/focus the
		typing surface first, then call this: each character fires a real keyDown+keyUp via CDP, so the
		page's keydown/keypress/input/keyup handlers all run. This is different from `ctx.type(index, …)`
		(which targets one element node and may set value) and `ctx.press(keys)` (a single key/shortcut).
		`per_char_delay` paces the typing — leave 0 to go as fast as possible, or set a small delay to
		throttle or to land near a target speed.
		"""
		if self._cancel.is_set():
			raise RuntimeError('Task stopped by the user.')
		cdp_session = await self._session.get_or_create_cdp_session(focus=True)
		session_id = cdp_session.session_id
		delay = max(0.0, min(float(per_char_delay), 1.0))
		for ch in text:
			if self._cancel.is_set():
				raise RuntimeError('Task stopped by the user.')
			down: dict[str, Any]
			up: dict[str, Any]
			if ch in ('\n', '\r'):
				down = {'type': 'keyDown', 'key': 'Enter', 'code': 'Enter', 'windowsVirtualKeyCode': 13, 'text': '\r'}
				up = {'type': 'keyUp', 'key': 'Enter', 'code': 'Enter', 'windowsVirtualKeyCode': 13}
			else:
				# Sending `text` on keyDown makes Chrome generate the character (keypress/input) like a real
				# keyboard; `key` carries the logical value so keydown handlers that read event.key still work.
				down = {'type': 'keyDown', 'key': ch, 'text': ch}
				up = {'type': 'keyUp', 'key': ch}
			await cdp_session.cdp_client.send.Input.dispatchKeyEvent(params=down, session_id=session_id)  # type: ignore[arg-type]
			await cdp_session.cdp_client.send.Input.dispatchKeyEvent(params=up, session_id=session_id)  # type: ignore[arg-type]
			if delay:
				await asyncio.sleep(delay)

	async def wait(self, seconds: float = 0.3) -> None:
		"""Pause between actions to control rhythm/timing."""
		await asyncio.sleep(max(0.0, min(float(seconds), 30.0)))

	async def js(self, script: str) -> Any:
		"""Run arbitrary JavaScript on the page and return its value.

		This is the helper's general escape hatch: read or paint canvas pixels, read page
		state, copy data between frames — anything the page's own JS can do — without
		round-tripping each step through the LLM.
		"""
		cdp_session = await self._session.get_or_create_cdp_session(focus=True)
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': script, 'returnByValue': True, 'awaitPromise': True},
			session_id=cdp_session.session_id,
		)
		if result and 'exceptionDetails' in result:
			details = result['exceptionDetails']
			exception = details.get('exception') or {}
			message = exception.get('description') or exception.get('value') or details.get('text') or 'JavaScript error'
			raise RuntimeError(str(message).split('\n')[0])
		return result['result'].get('value') if result and 'result' in result else None

	async def snapshot(self) -> Any:
		"""Return a compact fingerprint of every canvas: index, size, opaque-pixel count, a content
		hash, and on-screen rect.

		Call it BEFORE and AFTER a draw and compare: the canvas whose count/hash CHANGED is the one
		your action actually affected — the real editable surface. If nothing changed, your draw did
		not land (fix your input before continuing). This lets you verify by what YOU changed, not by
		the absolute page state — which already contains the source image and will fool an absolute check.
		"""
		return await self.js(
			'(() => Array.from(document.querySelectorAll("canvas")).map((c, i) => {'
			'  const r = c.getBoundingClientRect();'
			'  const base = {i, w: c.width, h: c.height, rect: {left: r.left, top: r.top, width: r.width, height: r.height}};'
			'  if (!c.width || !c.height) return {...base, count: 0, hash: 0};'
			'  let data;'
			'  try { data = c.getContext("2d", {willReadFrequently: true}).getImageData(0, 0, c.width, c.height).data; }'
			'  catch (e) { return {...base, count: 0, hash: 0, error: String(e)}; }'
			'  let count = 0, hash = 0;'
			'  for (let p = 0; p < data.length; p += 4) {'
			'    if (data[p + 3] > 10) { count++; hash = (hash * 31 + p + data[p] * 7 + data[p + 1] * 13 + data[p + 2] * 17 + data[p + 3]) >>> 0; }'
			'  }'
			'  return {...base, count, hash};'
			'}))()'
		)


class _ReplayHelperContext(_HelperContext):
	"""Runs a SAVED helper. The live indices it was written with are gone, so index-based clicks
	resolve through the stable selectors captured when the helper first worked. Coordinate/JS
	actions (click_at, drag, snapshot, js) are inherited unchanged."""

	def __init__(self, browser_session: BrowserSession, index_selectors: dict[str, str], cancel_event: asyncio.Event):
		super().__init__(browser_session, [], cancel_event)
		# JSON round-trips dict keys to strings; index args from the code are ints — match on str.
		self._saved_selectors = {str(key): value for key, value in (index_selectors or {}).items()}

	async def _selector_coords(self, index: int) -> tuple[float, float]:
		selector = self._saved_selectors.get(str(index))
		if not selector:
			raise ValueError(f'saved helper used index {index} but recorded no stable selector for it')
		coords = await self.js(
			'(() => {'
			f'  const s = {json.dumps(selector)};'
			'  let el;'
			'  if (s.startsWith("xpath:")) { const xp = s.slice(6); const p = xp.startsWith("/") ? xp : "/" + xp;'
			'    el = document.evaluate(p, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue; }'
			'  else { el = document.querySelector(s); }'
			'  if (!el) return null;'
			'  el.scrollIntoView({block: "center", inline: "center"});'
			'  const b = el.getBoundingClientRect();'
			'  return [b.left + b.width / 2, b.top + b.height / 2];'
			'})()'
		)
		if not coords:
			raise ValueError(f'saved selector {selector!r} is not on the page anymore')
		return float(coords[0]), float(coords[1])

	async def click(self, index: int, hold: float = 0.08) -> None:
		x, y = await self._selector_coords(index)
		await self.click_at(x, y, hold=hold)

	async def type(self, index: int, text: str, clear: bool = True) -> None:
		x, y = await self._selector_coords(index)
		await self.click_at(x, y)
		await self.js(
			'(() => { const el = document.activeElement; if (!el) return false;'
			f'  el.value = {json.dumps(text)};'
			'  el.dispatchEvent(new Event("input", {bubbles: true}));'
			'  el.dispatchEvent(new Event("change", {bubbles: true})); return true; })()'
		)


class PetBridge:
	"""Queue extension tasks and run them in the originating Chrome tab."""

	def __init__(self, cdp_url: str | None, max_steps: int, log_dir: Path, site_dir: Path):
		self.cdp_url = cdp_url
		self.max_steps = max_steps
		self.log_dir = log_dir
		self.site_dir = site_dir
		self.site_memory = PetSiteMemory(site_dir)
		self.trace_builder = PetTraceBuilder()
		self.skill_learner = PetSkillLearner(site_dir)
		self.skill_registry = PetSkillRegistry()
		self.sessions: dict[str, PetSession] = {}
		self._configure_session_logging()

	def _configure_session_logging(self) -> None:
		"""Keep terminal output readable and store detailed agent logs per tab session."""
		self.log_dir.mkdir(parents=True, exist_ok=True)
		session_handler = PetSessionLogHandler(self.log_dir)
		for logger_name in ('browser_use', 'bubus'):
			logger = logging.getLogger(logger_name)
			logger.handlers = [handler for handler in logger.handlers if not isinstance(handler, logging.StreamHandler)]
			if not any(isinstance(handler, PetSessionLogHandler) for handler in logger.handlers):
				logger.addHandler(session_handler)
			logger.setLevel(logging.INFO)
			logger.propagate = False

	def _get_session(self, session_id: str) -> PetSession:
		"""Return the state bucket for a deployed pet tab."""
		session = self.sessions.get(session_id)
		if session is None:
			session = PetSession(session_id=session_id)
			session.runner_task = asyncio.create_task(self._run_session_forever(session))
			self.sessions[session_id] = session
		return session

	def _session_from_request(self, request: web.Request) -> PetSession:
		"""Resolve a session id from query parameters for status-style requests."""
		session_id = request.query.get('session_id')
		if not session_id:
			raise ValueError('session_id is required')
		return self._get_session(session_id)

	def _task_hostname(self, task: PetTask) -> str:
		"""Return the normalized hostname for a submitted site-native task."""
		return self.site_memory.task_hostname(task.url)

	def _site_path(self, task: PetTask) -> Path:
		"""Return the local memory folder for the task's current site."""
		return self.site_memory.site_path(task.url)

	def _ensure_site_memory(self, task: PetTask) -> Path:
		"""Create a basic site memory file if this deployed site has none yet."""
		return self.site_memory.ensure_site_memory(task.url)

	def _site_memory_prompt(self, task: PetTask) -> str:
		"""Load reusable site notes, recent successes, and active skills for the agent prompt."""
		recent_traces = self._load_recent_site_traces(task, limit=3)
		active_skills = self.skill_registry.load_active_skills(self._site_path(task))
		active_skill_lines = self.skill_registry.prompt_lines(active_skills)
		return self.site_memory.prompt(task.url, recent_traces, active_skill_lines)

	async def _run_active_skill(
		self,
		task: PetTask,
		name: str,
		inputs: dict[str, Any],
		browser_session: BrowserSession,
	) -> ActionResult:
		"""Execute an active site skill — a saved live-helper is replayed here; others go to the registry."""
		site_path = self._site_path(task)
		skill = next((s for s in self.skill_registry.load_active_skills(site_path) if s.metadata.name == name), None)
		if skill is not None and skill.helper_path is not None:
			return await self._replay_helper_skill(skill, browser_session)
		return await self.skill_registry.execute(site_path, name, inputs, browser_session)

	async def _replay_helper_skill(self, skill: Any, browser_session: BrowserSession) -> ActionResult:
		"""Re-run a saved helper's verified code, resolving its old indices via stable selectors."""
		try:
			code = skill.helper_path.read_text(encoding='utf-8')
			index_selectors = (skill.metadata.capabilities or {}).get('index_selectors') or {}
			namespace: dict[str, Any] = {'ActionResult': ActionResult, 'asyncio': asyncio}
			exec(compile(code, '<saved_helper>', 'exec'), namespace)  # noqa: S102 — our own previously-verified helper
		except Exception as e:
			return ActionResult(error=f'Saved helper {skill.metadata.name!r} did not load: {e}')
		entrypoint = namespace.get('run')
		if not callable(entrypoint):
			return ActionResult(error=f'Saved helper {skill.metadata.name!r} has no run(ctx).')
		ctx = _ReplayHelperContext(browser_session, index_selectors, asyncio.Event())
		try:
			result = entrypoint(ctx)
			if inspect.isawaitable(result):
				result = await result
		except Exception as e:
			return ActionResult(error=f'Saved helper {skill.metadata.name!r} failed on replay: {e}')
		if isinstance(result, ActionResult):
			return result
		return ActionResult(extracted_content=f'Replayed saved helper ({ctx.click_count} actions).', include_in_memory=True)

	def _skill_save_decision(self, session: PetSession) -> Literal['helper', 'trace', 'none']:
		"""Choose how (or whether) to codify a verified-good run into a skill, from the run's SHAPE.

		The shape is read off the agent's own behavior, never hardcoded per site:
		- 'helper': a helper actually worked → save that verified code (handles real-time/freehand tasks).
		- 'none': the agent reached for code (it has helper attempts in attempt_log) but none worked →
		  a code-shaped task we could not codify this run. A trace re-derivation would emit a click/type
		  skill of the wrong shape that breaks on replay, so save nothing.
		- 'trace': the agent never reached for code → a plain click/type workflow the trace path fits.
		"""
		if any(helper.code for helper in session.authored_helpers):
			return 'helper'
		if session.attempt_log:
			return 'none'
		return 'trace'

	def _save_helper_skill(self, task: PetTask, session: PetSession) -> Path | None:
		"""Save the verified working helper as a reusable skill (its code + index->selector map),
		instead of re-deriving a new skill from the trace. The code is generic (it reads the live
		source and acts), so it reuses across inputs; the selector map lets its tool-clicks survive
		new page loads."""
		helpers = [helper for helper in session.authored_helpers if helper.code]
		if not helpers:
			return None
		helper = helpers[-1]  # the last authored helper produced the verified-good result
		site_path = self._site_path(task)
		skills_dir = site_path / 'skills'
		skills_dir.mkdir(parents=True, exist_ok=True)
		safe_name = re.sub(r'[^a-z0-9_]+', '_', task.task.lower()).strip('_')[:60] or 'saved_helper'
		metadata = {
			'name': safe_name,
			'status': 'trial',
			'version': 1,
			'description': f'Verified helper for: {task.task}',
			'inputs': [],
			'trigger': task.task,
			'entrypoint': 'helper',
			'capabilities': {'index_selectors': helper.index_selectors},
		}
		(skills_dir / f'{safe_name}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
		(skills_dir / f'{safe_name}.helper.py').write_text(helper.code, encoding='utf-8')
		self.skill_learner._register_in_manifest(site_path, safe_name)
		print(f'Saved verified helper as skill {safe_name!r} → {skills_dir / f"{safe_name}.helper.py"}')
		return skills_dir / f'{safe_name}.helper.py'

	async def _run_authored_helper(
		self,
		session: PetSession,
		code: str,
		element_indices: list[int],
		browser_session: BrowserSession,
	) -> ActionResult:
		"""Compile and run a helper the agent wrote live, grounded to declared element indices.

		The helper runs against real verified nodes (declared indices), so it executes the
		whole sequence in one shot without per-action LLM steps. On success we remember it on
		the session, along with the fingerprints of the elements it actually touched, so it can
		be saved as a reusable skill at the end of the run.
		"""
		code_hint = self._helper_code_hint(code)

		def finish(action: ActionResult, outcome: str) -> ActionResult:
			"""Record this attempt and ride the running history on the result, so the agent always
			sees what it already tried this run (even after old steps are trimmed) and stops repeating."""
			session.attempt_log.append(
				f'{len(session.attempt_log) + 1}. {code_hint} -> {self._clean_status_text(outcome, max_length=160)}'
			)
			history = 'Approaches tried this run (do NOT repeat a failed one — change the approach):\n' + '\n'.join(
				session.attempt_log[-12:]
			)
			body = action.error or action.extracted_content or ''
			combined = f'{body}\n\n{history}' if body else history
			if action.error:
				return ActionResult(error=combined)
			return ActionResult(extracted_content=combined, include_in_memory=True)

		namespace: dict[str, Any] = {'ActionResult': ActionResult, 'asyncio': asyncio}
		try:
			compiled = compile(code, '<authored_helper>', 'exec')
			exec(compiled, namespace)  # noqa: S102 — agent-authored helper, grounded to declared indices
		except SyntaxError as e:
			return finish(ActionResult(error=f'Helper did not compile: {e}'), f'compile error: {e}')
		entrypoint = namespace.get('run')
		if not callable(entrypoint):
			return finish(ActionResult(error='Helper must define `async def run(ctx):`.'), 'no run() defined')

		ctx = _HelperContext(browser_session, element_indices, session.cancel_event)
		try:
			result = entrypoint(ctx)
			if inspect.isawaitable(result):
				result = await result
		except Exception as e:
			return finish(ActionResult(error=f'Helper failed at runtime: {e}'), f'runtime error: {e}')

		session.authored_helpers.append(
			AuthoredHelper(
				code=code,
				element_indices=list(element_indices),
				used_elements=ctx.used_elements,
				index_selectors=dict(ctx.index_selectors),
			)
		)

		if isinstance(result, ActionResult):
			return finish(result, result.error or self._clean_status_text(result.extracted_content, max_length=160) or 'ok')
		# Report only clicks performed. Distinct-element counts mislead the agent: near-identical
		# targets (e.g. piano keys sharing tag/class with no id/aria/text) collapse to one
		# fingerprint, so "1 distinct element" looked like a failure even after dozens of real clicks.
		summary = f'Ran authored helper: performed {ctx.click_count} click(s) successfully.'
		if result is not None:
			summary += f' Result: {self._clean_status_text(result, max_length=200)}'
		outcome = f'{ctx.click_count} clicks; {self._clean_status_text(result, max_length=120) if result is not None else "no return value"}'
		return finish(ActionResult(extracted_content=summary, include_in_memory=True), outcome)

	async def _run_autopilot(
		self,
		session: PetSession,
		setup_js: str,
		max_seconds: float,
		browser_session: BrowserSession,
	) -> ActionResult:
		"""Run an LLM-authored game loop INSIDE the page at native speed, then report how it ended.

		This is the fast 'reflex' half of real-time play. `run_helper`'s loop lives in Python and
		round-trips to the page once PER move (tens of ms each) — fatal for a real-time game, since the
		world moves on while we wait, so the reflex acts on stale state (this is exactly why snake failed).
		Here the whole loop runs in the page on its own clock (rAF/setInterval): it reads state and
		dispatches keys locally, zero CDP latency per tick. Python only injects it once, polls a tiny
		result global, and hands back the final score + DEATH state so the LLM (the slow brain) can
		rewrite a smarter loop between deaths. The agent's contract: its JS must keep
		`window.__PET_AUTOPILOT = {done, score, ticks, death, error}` updated, set done=true when the game
		ends, and stop its loop when `window.__PET_AUTOPILOT_STOP` is true.
		"""
		cdp_session = await browser_session.get_or_create_cdp_session(focus=True)
		session_id = cdp_session.session_id
		# Reset the result + stop flag, then start the agent's loop. Its own errors are captured into the
		# result global (not raised) so a crashing reflex still reports back instead of vanishing.
		boot = (
			'window.__PET_AUTOPILOT_STOP = false;\n'
			'window.__PET_AUTOPILOT = {done: false, score: null, ticks: 0, death: null, error: null};\n'
			'try {\n'
			f'{setup_js}\n'
			'} catch (e) { const a = window.__PET_AUTOPILOT; a.error = String((e && e.stack) || e); a.done = true; }'
		)
		try:
			start = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': boot, 'returnByValue': True, 'awaitPromise': True},
				session_id=session_id,
			)
		except Exception as e:
			return self._finish_autopilot(session, f'autopilot failed to start: {e}', error=str(e))
		if start and 'exceptionDetails' in start:
			details = start['exceptionDetails']
			message = (details.get('exception') or {}).get('description') or details.get('text') or 'JavaScript error'
			return self._finish_autopilot(session, f'autopilot did not start: {message}', error=str(message))

		# Poll the in-page result while the reflex runs on its own. No LLM, no screenshots — just read a
		# small global. Stop when the loop reports done, the user cancels, or we hit the time budget.
		poll = 0.2
		max_polls = max(1, int(max(1.0, min(float(max_seconds), 120.0)) / poll))
		result: dict[str, Any] = {}
		for _ in range(max_polls):
			if session.cancel_event.is_set():
				break
			await asyncio.sleep(poll)
			read = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': 'window.__PET_AUTOPILOT', 'returnByValue': True},
				session_id=session_id,
			)
			value = (read.get('result') or {}).get('value') if read else None
			if isinstance(value, dict):
				result = value
				if value.get('done'):
					break
		# Always tell the loop to stop so it does not keep running into the next step.
		try:
			await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': 'window.__PET_AUTOPILOT_STOP = true', 'returnByValue': True},
				session_id=session_id,
			)
		except Exception:
			pass

		score = result.get('score')
		ticks = result.get('ticks')
		death = result.get('death')
		error = result.get('error')
		done = bool(result.get('done'))
		# The death state is the whole point: it is the lesson the LLM rewrites against. Surface it plainly.
		parts = [f'score={score}' if score is not None else 'score=unknown', f'ticks={ticks}' if ticks is not None else '']
		if not done:
			parts.append('(timed out — loop did not report done)')
		if death is not None:
			parts.append(
				f'DEATH: {self._clean_status_text(json.dumps(death) if isinstance(death, (dict, list)) else death, max_length=300)}'
			)
		if error:
			parts.append(f'ERROR: {self._clean_status_text(error, max_length=300)}')
		summary = 'Autopilot loop finished. ' + '; '.join(p for p in parts if p)
		return self._finish_autopilot(session, summary, error=error)

	def _finish_autopilot(self, session: PetSession, summary: str, error: str | None = None) -> ActionResult:
		"""Record an autopilot run on the session log and ride the running history, like helper attempts,
		so the LLM always sees what its past reflexes scored/how they died and stops repeating them."""
		session.attempt_log.append(
			f'{len(session.attempt_log) + 1}. autopilot -> {self._clean_status_text(summary, max_length=200)}'
		)
		history = 'Approaches tried this run (do NOT repeat a failed one — change the approach):\n' + '\n'.join(
			session.attempt_log[-12:]
		)
		combined = f'{summary}\n\n{history}'
		# A reflex that crashed at boot is an error; one that merely died in-game is normal feedback to learn from.
		if error and 'score=' not in summary:
			return ActionResult(error=combined)
		return ActionResult(extracted_content=combined, include_in_memory=True)

	def _helper_code_hint(self, code: str) -> str:
		"""A short label for a helper attempt: its first comment, else its first real line."""
		lines = [line.strip() for line in code.splitlines()]
		for line in lines:
			if line.startswith('#') and len(line) > 2:
				return self._clean_status_text(line.lstrip('# '), max_length=90)
		for line in lines:
			if line and not line.startswith(('async def run', 'def run', 'import ', 'from ', 'return await ctx.js')):
				return self._clean_status_text(line, max_length=90)
		return 'helper'

	def _load_recent_site_traces(self, task: PetTask, limit: int) -> list[dict[str, str]]:
		"""Return compact summaries from the most recent successful traces for this site."""
		trace_dir = self._site_path(task) / 'traces'
		if not trace_dir.exists():
			return []
		traces: list[dict[str, str]] = []
		for path in sorted(trace_dir.glob('*.json'), reverse=True):
			try:
				data = json.loads(path.read_text(encoding='utf-8'))
			except Exception:
				continue
			if not data.get('successful'):
				continue
			traces.append(
				{
					'task': self.trace_builder.clean_status_text(data.get('task', ''), max_length=220),
					'result': self._clean_status_text(
						data.get('final_result_summary') or data.get('final_result', ''), max_length=360
					),
				}
			)
			if len(traces) >= limit:
				break
		return traces

	def _save_site_trace(
		self,
		task: PetTask,
		session: PetSession,
		final_result: str | None,
		successful: bool,
		history: Any | None = None,
	) -> None:
		"""Persist a compact task trace so future runs on the same site have useful memory."""
		site_path = self._site_path(task)
		trace_dir = site_path / 'traces'
		trace_dir.mkdir(parents=True, exist_ok=True)
		now = datetime.now(UTC)
		trace_path = trace_dir / f'{now.strftime("%Y%m%d-%H%M%S")}-{task.id}.json'
		payload = {
			'version': 2,
			'timestamp': now.isoformat(),
			'session_id': session.session_id,
			'task_id': task.id,
			'url': task.url,
			'task': task.task,
			'successful': successful,
			'final_result_summary': self.trace_builder.final_result_summary(final_result),
		}
		operations = self.trace_builder.operations(task.task, history, session.visible_steps)
		if operations:
			payload['operations'] = [operation.model_dump() for operation in operations]
		trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

	async def _reflect_site_memory(
		self,
		task: PetTask,
		llm: ChatOpenClaw,
		final_result: str | None,
		completed_items: list[str],
	) -> SiteMemoryReflection:
		"""Ask the LLM for reusable website lessons."""
		return await self.site_memory.reflect(task.url, task.task, llm, final_result, completed_items)

	def _append_site_memory_notes(self, task: PetTask, notes: list[str]) -> list[str]:
		"""Append non-duplicate memory notes to this site's memory file."""
		return self.site_memory.append_notes(task.url, notes)

	def _site_memory_notes(self, memory: str) -> list[str]:
		"""Extract bullet notes from a site memory file."""
		return self.site_memory.site_memory_notes(memory)

	def _should_reflect_site_memory(self, history: Any | None) -> bool:
		"""Return whether a successful run had enough friction to learn a site lesson."""
		return self.site_memory.should_reflect(history)

	async def submit(self, request: web.Request) -> web.Response:
		"""Validate and queue a browser task."""
		try:
			task = PetTask.model_validate(await request.json())
		except Exception as e:
			return self._json_response({'ok': False, 'error': str(e)}, status=400)

		session = self._get_session(task.session_id)
		await session.queue.put(task)
		if session.current_agent is None:
			session.status = PetStatus(state='queued', message='Task queued', task_id=task.id)
		return self._json_response({'ok': True, 'task_id': task.id})

	async def options(self, request: web.Request) -> web.Response:
		"""Allow the unpacked extension to call the local bridge."""
		return self._json_response({'ok': True})

	async def health(self, request: web.Request) -> web.Response:
		"""Return bridge health for extension diagnostics."""
		return self._json_response({'ok': True})

	async def get_status(self, request: web.Request) -> web.Response:
		"""Return the current agent status for the visible extension panel."""
		try:
			session = self._session_from_request(request)
		except Exception as e:
			return self._json_response({'ok': False, 'error': str(e)}, status=400)
		return self._json_response(session.status.model_dump())

	async def reply(self, request: web.Request) -> web.Response:
		"""Deliver a user reply to the active ask_human action."""
		try:
			reply = PetReply.model_validate(await request.json())
		except Exception as e:
			return self._json_response({'ok': False, 'error': str(e)}, status=400)
		session = self._get_session(reply.session_id)
		if session.status.state != 'waiting':
			return self._json_response({'ok': False, 'error': 'The pet is not waiting for a reply.'}, status=409)

		await session.reply_queue.put(reply.reply)
		return self._json_response({'ok': True})

	async def stop(self, request: web.Request) -> web.Response:
		"""Request cancellation of the active task without shutting down the bridge."""
		try:
			session = self._session_from_request(request)
		except Exception as e:
			return self._json_response({'ok': False, 'error': str(e)}, status=400)
		if session.status.state not in {'queued', 'running', 'waiting', 'stopping'}:
			return self._json_response({'ok': True})

		session.status = PetStatus(state='stopping', message='Stopping task...', task_id=session.status.task_id)
		if session.status.task_id is not None:
			session.cancelled_task_ids.add(session.status.task_id)
		session.cancel_event.set()
		if session.current_agent is not None:
			session.current_agent.stop()
		return self._json_response({'ok': True})

	async def run_forever(self) -> None:
		"""Keep the bridge process alive while per-tab sessions do the work."""
		await asyncio.Event().wait()

	async def _run_session_forever(self, session: PetSession) -> None:
		"""Process tasks sequentially inside one deployed pet tab."""
		while True:
			task = await session.queue.get()
			try:
				if task.id in session.cancelled_task_ids:
					session.status = PetStatus(state='stopped', message='Task stopped', task_id=task.id)
				else:
					token = _PET_LOG_SESSION.set(session.session_id)
					try:
						await self._run_task(task, session)
					finally:
						_PET_LOG_SESSION.reset(token)
			except Exception as e:
				print(f'Pet task failed in {session.session_id}: {e}')
				self._save_site_trace(task, session, f'Task failed: {e}', successful=False)
				session.status = PetStatus(state='failed', message=f'Task failed: {e}', task_id=task.id)
			finally:
				session.current_agent = None
				session.queue.task_done()

	async def _run_task(self, task: PetTask, session: PetSession) -> None:
		"""Attach browser-use to Chrome and run a task in its originating tab."""
		session.cancel_event = asyncio.Event()
		session.reply_queue = asyncio.Queue()
		session.visible_steps = []
		session.authored_helpers = []
		session.attempt_log = []
		session.notes = []
		session.status = PetStatus(state='running', message='Planning task...', task_id=task.id)
		cdp_url = self.cdp_url or discover_chrome_cdp_url()
		browser = Browser(
			browser_profile=BrowserProfile(
				cdp_url=cdp_url,
				keep_alive=True,
				pet_mode=True,
				pet_origin_token=task.origin_token,
			)
		)
		await browser.start()
		origin_target_id = await self._focus_originating_tab(browser, task)

		# Pristine pre-task screenshot. The end-of-run visual check compares the RESULT against this
		# reference (result-vs-reference), instead of judging the final screenshot against the task
		# prose alone — that prose-only check rubber-stamps anything that "looks done" (e.g. a redraw
		# with the wrong colors). For copy/reproduce/match tasks the reference IS this before-state.
		before_image = await self._safe_screenshot(browser)

		print(
			f'Running pet task [{session.session_id}] on {task.url}: {task.task}\n'
			f'  log: {self.log_dir / f"{_safe_log_name(session.session_id)}.log"}'
		)
		tools = self._create_tools(task, session)
		# Pet tasks include freehand work on <canvas> (drawing, signatures) where index-based
		# clicks only ever hit element centers. Coordinate clicking lets the LLM aim at (x, y).
		tools.set_coordinate_clicking(True)
		llm = _pet_llm()
		contract = await self._create_execution_contract(task, llm)
		site_memory = self._site_memory_prompt(task)
		total_units = contract.total_units if contract.mode == 'sequential' else 1
		last_history = None
		completed_items: list[str] = []

		async def should_stop() -> bool:
			return session.cancel_event.is_set()

		async def on_new_step(browser_state_summary: Any, model_output: Any, step_number: int) -> None:
			await self._bind_browser_to_origin_target(browser, origin_target_id)
			step_status = self._format_step_status(model_output, step_number)
			session.visible_steps.append(step_status)
			session.status = PetStatus(
				state='running',
				message=step_status,
				task_id=task.id,
			)

		def build_unit_task(unit_number: int) -> tuple[str, str]:
			if contract.mode == 'sequential':
				completed_context = ''
				if completed_items:
					completed_context = (
						'\n\nCompleted source items so far. Treat these as blocked and do not act on them again:\n'
						+ '\n'.join(f'{index}. {item}' for index, item in enumerate(completed_items, start=1))
					)
				unit_task = (
					f'{contract.unit_task}\n\n'
					f'You are completing item {unit_number} of {total_units}. Handle exactly one item. '
					'Do not inspect, prepare, or mention later items. '
					'This item must use a distinct source item that has not already been completed in this same session. '
					'Before acting, identify the current source item by its visible title, author, URL, or other stable page text. '
					'If the page is still focused on or showing a source item already completed, leave that item, close its composer or modal if needed, scroll or navigate to a different item, then observe again. '
					'Never reply to, react to, apply to, message about, or otherwise act on the same source item twice unless the user explicitly asks for duplicates. '
					'Stay in the originating browser tab unless the user explicitly requested a tab switch or navigation to another website. '
					'Prepare the current item inside the website UI before asking for approval. '
					'This unit is complete when the current single item has been completed or skipped. '
					'Do not attempt to complete the overall multi-item request inside this unit.'
					f'{completed_context}'
				)
				status_message = f'Working on item {unit_number} of {total_units}...'
			else:
				unit_task = (
					f'{task.task}\n\n'
					'If this is a low-risk repeated reaction task, use react_to_visible_items with the requested reaction '
					'and count. Choose eligible visible items yourself, scroll only when needed, and call done only once '
					'after the full requested count is complete.'
				)
				status_message = 'Working...'
			return unit_task, status_message

		first_unit_task, _ = build_unit_task(1)
		agent = Agent(
			task=first_unit_task + site_memory + _SAFETY_INSTRUCTIONS,
			llm=llm,
			browser=browser,
			tools=tools,
			# True = the agent sees a screenshot each step. Drawing/canvas work blind was the core failure:
			# it computed pixel→screen coordinates, clicked, missed, and could not tell (it had no eyes,
			# only self-written verify code it got wrong). Letting it LOOK after a draw lets it notice its
			# own misses and self-correct — like a person would — which is how it reaches a first success
			# the memory loop can then capture.
			use_vision=True,
			# We enable coordinate clicking/drag above, so the model reads pixel coordinates off the
			# screenshot. Pin a fixed screenshot size so those coordinates map back to CSS viewport pixels
			# exactly (otherwise on a retina display the model sees a 2x image and every click/drag lands
			# at double the intended position, missing the canvas entirely).
			llm_screenshot_size=(1400, 850),
			use_judge=False,
			llm_timeout=_PET_LLM_TIMEOUT_SECONDS,
			step_timeout=_PET_STEP_TIMEOUT_SECONDS,
			directly_open_url=False,
			register_new_step_callback=on_new_step,
			register_should_stop_callback=should_stop,
			save_conversation_path='/tmp/pet_conversation.txt',
		)
		session.current_agent = agent

		for unit_number in range(1, total_units + 1):
			if session.cancel_event.is_set():
				session.status = PetStatus(state='stopped', message='Task stopped', task_id=task.id)
				return
			if contract.mode == 'sequential' and last_history is not None:
				completed_items.append(self._summarize_completed_item(last_history.final_result(), unit_number))

			unit_task, status_message = build_unit_task(unit_number)
			session.status = PetStatus(state='running', message=status_message, task_id=task.id)
			if unit_number > 1:
				agent.add_new_task(unit_task + site_memory + _SAFETY_INSTRUCTIONS)
			await self._bind_browser_to_origin_target(browser, origin_target_id)
			last_history = await agent.run(max_steps=self.max_steps)
			if session.cancel_event.is_set():
				session.status = PetStatus(state='stopped', message='Task stopped', task_id=task.id)
				return

			# Learn incrementally on repeated-item runs. An open-ended task ("apply to jobs until I
			# stop / until credits run out") never reaches the end-of-run learning below, so learn
			# from each completed item. Gated on "no active skill yet" → at most one proposal call
			# per workflow, self-healing if a bad skill was rolled back. Then refresh site_memory so
			# the NEXT unit can actually use what we just learned.
			if (
				contract.mode == 'sequential'
				and last_history is not None
				and not self.skill_registry.load_active_skills(self._site_path(task))
			):
				self._save_site_trace(task, session, last_history.final_result(), successful=True, history=last_history)
				await self._propose_site_skill(task, llm)
				site_memory = self._site_memory_prompt(task)

		final_result = last_history.final_result() if last_history else None

		# Self-improving gate: the agent's own in-code "done" can be vacuous, so an independent signal
		# checks whether the task is REALLY done, and if not, the agent adapts and retries — in-run
		# self-correction instead of giving up after one shot. The signal is chosen PER TASK, not
		# hardcoded: first we ask the agent to define its own numeric scoreboard (a JS metric it reads
		# off the page, e.g. the max 2048 tile). If it can, we optimize that number (keep going while it
		# climbs, change strategy when it stalls, stop at the goal). If the task can't be a number (e.g.
		# "make it look like this"), we fall back to the visual judge — now just one kind of signal.
		verified_ok = True
		if last_history is not None and not session.cancel_event.is_set():
			plan = await self._create_progress_plan(task, browser, llm)
			if plan is not None:
				verified_ok, last_history, final_result = await self._run_progress_loop(
					task, session, agent, browser, plan, origin_target_id, site_memory, final_result, last_history
				)
			else:
				verified_ok, last_history, final_result = await self._run_visual_fix_loop(
					task, session, agent, browser, llm, before_image, origin_target_id, site_memory, final_result, last_history
				)

		session.status = PetStatus(
			state='completed' if verified_ok else 'failed',
			message=final_result or ('Task completed' if verified_ok else 'Could not verify the task was completed'),
			task_id=task.id,
		)
		self._save_site_trace(task, session, final_result, successful=verified_ok, history=last_history)
		# Learn only from a verified-good run, and only via a path matched to the run's SHAPE — which the
		# run itself reveals through the agent's own choice of tools, so we never hardcode "this site is a
		# game / that one is a form":
		#   - a helper actually WORKED  -> code-shaped task; save that verified helper (it just ran).
		#   - the agent reached for code but none of it worked (attempt_log non-empty) -> still code-shaped,
		#     just not codified this run; re-deriving a click/type skill from the trace would be the WRONG
		#     shape and break on replay (e.g. the monkeytype typing test, where the trace skill typed via
		#     .value and stalled), so we save NOTHING rather than an unverified guess.
		#   - the agent never touched code -> a plain click/type workflow; the trace->skill path fits.
		if verified_ok:
			decision = self._skill_save_decision(session)
			if decision == 'helper':
				self._save_helper_skill(task, session)
			elif decision == 'trace':
				await self._propose_site_skill(task, llm)
			else:
				print(
					f'Code-shaped run with no working helper for {self._task_hostname(task)}; '
					'saving no skill (trace re-derivation would be the wrong shape).'
				)
		if self._should_reflect_site_memory(last_history):
			reflection = await self._reflect_site_memory(task, llm, final_result, completed_items)
			removed_notes = self.site_memory.remove_notes(task.url, reflection.outdated_notes)
			if removed_notes:
				print(f'Removed {len(removed_notes)} stale site memory note(s) for {self._task_hostname(task)}')
			learned_notes = self._append_site_memory_notes(task, reflection.notes)
			if learned_notes:
				print(f'Learned {len(learned_notes)} site memory note(s) for {self._task_hostname(task)}')
			else:
				print(f'No high-value site memory notes learned for {self._task_hostname(task)}.')
		else:
			print(f'Skipped site memory reflection for {self._task_hostname(task)}; run had low friction.')

	async def _safe_screenshot(self, browser: Browser) -> str | None:
		"""Take a screenshot as a base64 data URL, or None if it fails.

		Verification degrades gracefully rather than crashing the run: a missing reference just
		means the visual check falls back to judging the result on its own.
		"""
		try:
			png = await browser.take_screenshot()
			return f'data:image/png;base64,{base64.b64encode(png).decode()}'
		except Exception as e:
			print(f'Could not capture screenshot: {e}')
			return None

	async def _visual_verify(
		self, task: PetTask, browser: Browser, llm: ChatOpenClaw, before_image: str | None = None
	) -> CompletionVerdict:
		"""Visual ground-truth check that the task is actually done — judged against a reference.

		The agent's own in-code verdict is self-graded and can be vacuously true (it checks "did the
		canvas change", not "is the result correct"). So the real arbiter is an independent look at
		the rendered result, compared against the pre-task reference (`before_image`) — NOT against
		the task prose alone, which rubber-stamps anything that merely "looks done". For a
		copy/reproduce/match task the before-state IS the thing the result must match, so wrong
		colors/shapes/missing parts now fail. Vision is used ONLY here, so the agent can't lean on
		it to do the work, only to confirm it. Fails CLOSED: any error => achieved=False.
		"""
		try:
			after_image = await self._safe_screenshot(browser)
			if after_image is None:
				return CompletionVerdict(achieved=False, reason='visual check could not capture a screenshot')

			content: list[Any] = [ContentPartTextParam(text=f'Task: {task.task}')]
			if before_image is not None:
				content.append(ContentPartTextParam(text='BEFORE — the page state before the agent acted (the reference):'))
				content.append(ContentPartImageParam(image_url=ImageURL(url=before_image)))
				content.append(ContentPartTextParam(text='AFTER — the result of the attempt:'))
			content.append(ContentPartImageParam(image_url=ImageURL(url=after_image)))
			content.append(ContentPartTextParam(text='Did the agent actually complete the task correctly?'))

			response = await llm.ainvoke(
				[
					SystemMessage(
						content=(
							'You are a strict QA checker for a browser agent. You are given an AFTER screenshot of the '
							'result, and usually a BEFORE screenshot of the page before the agent acted. Look carefully '
							'at the WHOLE image — including any thumbnail/frame strip, list, or preview that shows the '
							'result — and decide if the task was ACTUALLY completed and looks correct.\n'
							'If the task was to reproduce, copy, match, redraw, or recreate something, the AFTER result '
							'must CLOSELY match its reference (shown in BEFORE, or elsewhere on screen such as a source '
							'frame/thumbnail): wrong colors, wrong shapes, missing or extra parts, or partial/garbled '
							'output is achieved=false. Comparing identical pixels is not required — judge whether a '
							'person would accept it as a faithful match.\n'
							'A blank/empty result, or one that does not match what was asked, is achieved=false. '
							'If you cannot clearly confirm a correct result, return achieved=false. Give a one-line reason.\n'
							'Also set match = 0-100: how close the result is to fully correct (100 = perfect, 0 = nothing/'
							'totally wrong). Be honest and consistent so repeated checks can tell whether it is improving.'
						)
					),
					UserMessage(content=content),
				],
				output_format=CompletionVerdict,
			)
			return response.completion
		except Exception as e:
			# Fail CLOSED: if we cannot actually verify, treat it as NOT done. Better to save
			# nothing (and report unverified) than to rubber-stamp + save a wrong skill.
			print(f'Visual verify failed for {self._task_hostname(task)}: {e}')
			return CompletionVerdict(achieved=False, reason=f'visual check could not run: {e}')

	async def _create_progress_plan(self, task: PetTask, browser: Browser, llm: ChatOpenClaw) -> ProgressPlan | None:
		"""Ask the agent to define its OWN numeric scoreboard for this task.

		This is the generalization of the Piskel-specific visual check: rather than us picking the
		success signal, the LLM looks at the live page and writes a JS expression that reads progress
		as a single number (higher = closer) plus the value that means done. We hand it the current
		page's text/structure so the metric references things that actually exist. If it can't reduce
		the task to a number it returns usable=False and we use the visual judge instead. Fails soft:
		any error => None (caller falls back to visual)."""
		try:
			# A compact, side-effect-free read of the page so the metric the LLM writes refers to real
			# on-page text/numbers (not invented selectors). Title + visible body text, truncated.
			page_text = await self._page_text_snapshot(browser)
			response = await llm.ainvoke(
				[
					SystemMessage(
						content=(
							'You design a numeric SCOREBOARD for a browser task so an agent can tell, for free, '
							'whether it is getting closer to done. Given the task and a snapshot of the live page, '
							'decide if progress can be read as a single NUMBER off the page where higher = closer to '
							'done.\n'
							'- If yes: set usable=true, and write metric_js: a pure, side-effect-free JavaScript '
							'EXPRESSION (no statements, no assignment) that evaluates to that number by reading the '
							'DOM. It must not click, type, or change anything. Example for the game 2048: the current '
							'maximum tile value on the board. Set goal to the value that means the task is complete '
							'(e.g. 512), metric_label to a short human name, and strategy to a concrete first approach '
							'in plain words.\n'
							'- If the task cannot be honestly reduced to a single page-readable number (e.g. "make the '
							'drawing look like this picture", "write a nice reply"), set usable=false and leave the '
							'rest empty.\n'
							'Never invent elements: only reference things visible in the page snapshot. Prefer a robust '
							'read (e.g. scan all tiles and take the max) over a brittle single selector.'
						)
					),
					UserMessage(content=f'Task: {task.task}\nURL: {task.url}\n\nLive page snapshot:\n{page_text}'),
				],
				output_format=ProgressPlan,
			)
			plan = response.completion
			if not plan.usable or not plan.metric_js.strip():
				return None
			# Sanity-check the metric actually reads a number on THIS page before trusting it; a metric
			# that throws or returns non-numeric is worthless and would stall the loop on a fake signal.
			probe = await self._read_progress(browser, plan.metric_js)
			if probe is None:
				print(f'Progress metric did not return a number on {self._task_hostname(task)}; using visual check.')
				return None
			print(
				f'Progress scoreboard for {self._task_hostname(task)}: {plan.metric_label or plan.metric_js} (now {probe}, goal {plan.goal}).'
			)
			return plan
		except Exception as e:
			print(f'Could not build a progress scoreboard for {self._task_hostname(task)}: {e}; using visual check.')
			return None

	async def _page_text_snapshot(self, browser: Browser, max_chars: int = 6000) -> str:
		"""Compact, side-effect-free read of the page (title + visible text), for metric design."""
		try:
			cdp_session = await browser.get_or_create_cdp_session(focus=True)
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={
					'expression': '(() => (document.title + "\\n" + (document.body ? document.body.innerText : "")).slice(0, 12000))()',
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)
			text = (result.get('result') or {}).get('value') if result else None
			return str(text or '')[:max_chars]
		except Exception:
			return ''

	async def _read_progress(self, browser: Browser, metric_js: str) -> float | None:
		"""Evaluate the agent's progress metric and coerce it to a float — None if it can't.

		This is the cheap heartbeat of the self-improving loop: pure JS, no LLM, no screenshot, so the
		orchestrator can read the score between (and the agent's helper can read it during) every round."""
		try:
			cdp_session = await browser.get_or_create_cdp_session(focus=True)
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': f'(() => {{ return ({metric_js}); }})()', 'returnByValue': True, 'awaitPromise': True},
				session_id=cdp_session.session_id,
			)
			if not result or 'exceptionDetails' in result:
				return None
			value = (result.get('result') or {}).get('value')
			if isinstance(value, bool) or value is None:
				return None
			return float(value)
		except Exception:
			return None

	async def _run_progress_loop(
		self,
		task: PetTask,
		session: PetSession,
		agent: Agent,
		browser: Browser,
		plan: ProgressPlan,
		origin_target_id: Any,
		site_memory: str,
		final_result: str | None,
		last_history: Any,
	) -> tuple[bool, Any, str | None]:
		"""Optimize the pet's self-defined numeric metric: act → read score → keep/adapt → repeat.

		This is the general self-prompting engine. Each round the orchestrator reads the metric for free
		(pure JS, no LLM), then prompts the AGENT with the live number, the goal, and whether to keep the
		working approach or switch strategy because it stalled. The agent does the heavy lifting fast in
		code (one helper that loops many moves and self-checks the metric), so reaching e.g. tile 512 is a
		handful of LLM calls, not hundreds. Done = metric reaches goal; give up = metric stops climbing.
		"""
		goal = plan.goal
		label = plan.metric_label or plan.metric_js
		last = await self._read_progress(browser, plan.metric_js)
		best = last if last is not None else float('-inf')
		strategy = plan.strategy
		no_improvement = 0
		verified_ok = last is not None and last >= goal
		if verified_ok:
			print(f'Progress goal already met: {label} = {last} >= {goal}.')
		for attempt in range(_MAX_FIX_ATTEMPTS):
			if verified_ok or session.cancel_event.is_set():
				break
			reached = 'unknown' if last is None else _format_metric(last)
			# If a full agent run did not beat the best score, the current approach is exhausted —
			# tell it to genuinely CHANGE strategy rather than grind the same one.
			if no_improvement >= 1:
				guidance = (
					f'Your score is stuck around {_format_metric(best)} and the current approach is not breaking '
					'through. Switch to a genuinely DIFFERENT strategy (a different rule, order, or technique) and try again.'
				)
			else:
				guidance = 'The approach is working — keep going and push the score higher toward the goal.'
			self_prompt = (
				f'{task.task}\n\n'
				f'You are improving toward a measurable goal. Progress is measured by this exact JavaScript, '
				f'which returns a number where higher = better: `{plan.metric_js}` — it is {reached} now, goal {_format_metric(goal)} '
				f'({label}).\n'
				f'Strategy to use now: {strategy}\n'
				f'{guidance}\n'
				'Do the work FAST in code: write ONE run_helper that performs many moves/actions in a tight loop and '
				'reads this SAME metric via ctx.js between moves, continuing until the metric reaches the goal or clearly '
				'stops increasing, then returns the final metric value. Use your eyes to confirm the right controls and '
				'that it is actually working, but do NOT relay every move back to me one at a time.'
				+ site_memory
				+ _SAFETY_INSTRUCTIONS
			)
			session.status = PetStatus(
				state='running', message=f'Improving ({label}: {reached}/{_format_metric(goal)})...', task_id=task.id
			)
			agent.add_new_task(self_prompt)
			await self._bind_browser_to_origin_target(browser, origin_target_id)
			last_history = await agent.run(max_steps=self.max_steps)
			final_result = last_history.final_result() if last_history else final_result
			new = await self._read_progress(browser, plan.metric_js)
			if new is not None and new >= goal:
				verified_ok = True
				last = new
				print(f'Progress goal reached: {label} = {_format_metric(new)} >= {_format_metric(goal)}.')
				break
			if new is not None and new > best:
				best = new
				no_improvement = 0
				print(
					f'Progress climbing: {label} = {_format_metric(new)} (best {_format_metric(best)}, goal {_format_metric(goal)}).'
				)
			else:
				no_improvement += 1
				print(
					f'Progress did not improve ({label} ~{_format_metric(best)}); attempt {attempt + 1}, stalls {no_improvement}.'
				)
				# Three full runs with no new best means the agent is out of ideas for this task — stop.
				if no_improvement >= 3:
					print(f'Progress stuck at {_format_metric(best)}/{_format_metric(goal)} ({label}); stopping retries.')
					break
			last = new if new is not None else last
		return verified_ok, last_history, final_result

	async def _run_visual_fix_loop(
		self,
		task: PetTask,
		session: PetSession,
		agent: Agent,
		browser: Browser,
		llm: ChatOpenClaw,
		before_image: str | None,
		origin_target_id: Any,
		site_memory: str,
		final_result: str | None,
		last_history: Any,
	) -> tuple[bool, Any, str | None]:
		"""Visual self-correction for tasks with no numeric metric (copy/reproduce/match this picture).

		Independent judge LOOKS and compares the result to the before-state; if not done, feeds the
		specific complaint back and lets the agent fix it, then re-judges. Stops when it passes or the
		match score stalls. This is the old Piskel path — now just the fallback when no scoreboard fits.
		"""
		verified_ok = True
		best_match = -1
		no_improvement = 0
		for attempt in range(_MAX_FIX_ATTEMPTS):
			verdict = await self._visual_verify(task, browser, llm, before_image)
			verified_ok = verdict.achieved
			if verified_ok or session.cancel_event.is_set():
				break
			# Two non-improving tries means it's spinning, so we don't burn the whole budget repeating it.
			if verdict.match > best_match:
				best_match = verdict.match
				no_improvement = 0
			else:
				no_improvement += 1
				if no_improvement >= 2:
					print(f'Visual check stuck at ~{best_match}% match ({verdict.reason}); stopping retries.')
					break
			print(f'Visual check NOT done (~{verdict.match}% match: {verdict.reason}); fix attempt {attempt + 1}.')
			session.status = PetStatus(state='running', message='Fixing the result...', task_id=task.id)
			agent.add_new_task(
				f'{task.task}\n\nA visual check of the result said it is NOT actually done '
				f'(only ~{verdict.match}% match): {verdict.reason}. '
				'Look at the real rendered result, fix it so it genuinely matches what was asked, then finish.'
				+ site_memory
				+ _SAFETY_INSTRUCTIONS
			)
			await self._bind_browser_to_origin_target(browser, origin_target_id)
			last_history = await agent.run(max_steps=self.max_steps)
			final_result = last_history.final_result() if last_history else final_result
		return verified_ok, last_history, final_result

	async def _propose_site_skill(self, task: PetTask, llm: ChatOpenClaw) -> None:
		"""Propose and auto-generate a reusable site skill from recent successful traces.

		Default path generates a parameterized Python skill grounded in the elements the run
		actually clicked (recorded in the verified trace), so a repeatable workflow like the
		megamillions date filter becomes instant next time. The old steps generator (which had
		the LLM re-guess selectors from trace text) is gated behind a flag for comparison.
		"""
		hostname = self._task_hostname(task)
		# Show the proposer what already exists so it doesn't coin a new name for a skill we
		# already have (which is how dups like play_*_melody vs play_*_sequence appeared).
		existing_skills = self.skill_registry.load_active_skills(self._site_path(task))
		existing_skill_lines = self.skill_registry.prompt_lines(existing_skills)
		try:
			proposal = await self.skill_learner.propose(hostname, llm, existing_skills=existing_skill_lines)
		except Exception as e:
			print(f'Skipped skill proposal for {hostname}: {e}')
			return
		if proposal.trace_count < 1:
			return
		if not proposal.proposal.worth_building or not proposal.proposal.name:
			print(f'No skill worth building for {hostname}')
			return
		try:
			if _STEPS_SKILL_GENERATION_ENABLED:
				skill_path = await self.skill_learner.generate_skill_steps(proposal, llm)
			else:
				skill_path = await self.skill_learner.generate_skill_code(proposal, llm)
			print(f'Generated skill {proposal.proposal.name!r} for {hostname} → {skill_path}')
		except Exception as e:
			print(f'Skill generation failed for {hostname}: {e}')

	async def _create_execution_contract(self, task: PetTask, llm: ChatOpenClaw) -> ExecutionContract:
		"""Parse a user request into a single task or a repeated-item execution plan."""
		try:
			response = await llm.ainvoke(
				[
					SystemMessage(content=_TASK_PARSER_SYSTEM_PROMPT),
					UserMessage(content=f'Originating tab URL: {task.url}\nUser request: {task.task}'),
				],
				output_format=ExecutionContract,
			)
			contract = response.completion
			if contract.mode == 'single':
				return contract.model_copy(update={'unit_task': task.task, 'total_units': 1})
			if self._contains_unrequested_domain(contract.unit_task, task):
				print(f'Pet task parser introduced an unrequested domain; using current-tab unit task: {contract.unit_task}')
				return contract.model_copy(update={'unit_task': self._current_tab_unit_task(task)})
			return contract.model_copy(update={'unit_task': self._sanitize_unit_task(contract.unit_task, task)})
		except Exception as e:
			print(f'Pet task parser failed, running task unchanged: {e}')
			return ExecutionContract(
				mode='single',
				unit_task=task.task,
				total_units=1,
				completion_condition='The requested browser task is complete.',
			)

	def _current_tab_unit_task(self, task: PetTask) -> str:
		"""Build a generic unit task that stays on the originating tab."""
		return (
			f'On the current website/tab, complete exactly one item from this request: "{task.task}". '
			'Use the current page and same-site navigation/search only. '
			'Do not navigate to another website or job board unless the user explicitly requested that website. '
			'If the current page cannot support the request, ask the human what page or section to use.'
		)

	def _sanitize_unit_task(self, unit_task: str, task: PetTask) -> str:
		"""Remove parser-added workflow expansions that change the user's apply intent."""
		if not self._is_quick_apply_request(task.task):
			return unit_task
		for phrase in (
			'Read the job description and requirements carefully.',
			'Review the description to determine if you are a reasonable fit.',
			'If the job genuinely matches and you are confident proceeding, ',
			'If the job genuinely matches and you are confident proceeding,',
			'determine if you are a reasonable fit. ',
			'read the job description and requirements carefully. ',
		):
			unit_task = unit_task.replace(phrase, '')
		return f'{unit_task.strip()}\n\n{_QUICK_APPLY_INSTRUCTION}'

	def _is_quick_apply_request(self, task: str) -> bool:
		"""Detect apply tasks where the user requested the quick-apply workflow."""
		task_lower = task.lower()
		return 'apply' in task_lower and any(marker in task_lower for marker in ('easy apply', 'quick apply'))

	def _contains_unrequested_domain(self, generated_task: str, task: PetTask) -> bool:
		"""Detect parser hallucinations that add domains the user did not request."""
		generated_domains = self._extract_domains(generated_task)
		if not generated_domains:
			return False
		allowed_domains = self._extract_domains(task.task)
		current_domain = urlparse(task.url).hostname
		if current_domain:
			allowed_domains.add(current_domain.removeprefix('www.').lower())
		return any(domain not in allowed_domains for domain in generated_domains)

	def _extract_domains(self, text: str) -> set[str]:
		"""Extract explicit domain-like strings from text."""
		domains: set[str] = set()
		for match in re.finditer(r'https?://[^\s)>\]]+|www\.[^\s)>\]]+', text, flags=re.IGNORECASE):
			host = urlparse(match.group(0) if match.group(0).startswith('http') else f'https://{match.group(0)}').hostname
			if host:
				domains.add(host.removeprefix('www.').lower())
		for match in re.finditer(r'\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+\b', text, flags=re.IGNORECASE):
			domain = match.group(0).removeprefix('www.').lower()
			if not domain.endswith(('.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg')):
				domains.add(domain)
		return domains

	def _summarize_completed_item(self, final_result: str | None, unit_number: int) -> str:
		"""Create a compact completed-item identifier for repeated task follow-ups."""
		if not final_result:
			return f'Item {unit_number} completed.'
		summary = ' '.join(final_result.split())
		return summary[:700]

	def _format_step_status(self, model_output: Any, step_number: int) -> str:
		"""Format a model step into a short visible pet status."""
		lines = [f'Step {max(step_number, 1)}']
		evaluation = self._clean_status_text(getattr(model_output, 'evaluation_previous_goal', None))
		next_goal = self._clean_status_text(getattr(model_output, 'next_goal', None))
		memory = self._clean_status_text(getattr(model_output, 'memory', None))
		if evaluation and not evaluation.lower().startswith('this is the first step'):
			lines.append(f'Last: {evaluation}')
		if next_goal:
			lines.append(f'Now: {next_goal}')
		elif memory:
			lines.append(f'Now: {memory}')
		action_list = list(getattr(model_output, 'action', None) or [])
		actions = self._format_step_actions(action_list)
		if actions:
			lines.append(f'ActionCount: {len(action_list)}')
			lines.append(f'Action: {actions}')
		return '\n'.join(lines)

	def _format_step_actions(self, action_list: list) -> str:
		"""Create a compact action summary for the visible pet bubble."""
		if not action_list:
			return ''
		labels: list[str] = []
		for action in action_list[:3]:
			data = action.model_dump(exclude_none=True) if hasattr(action, 'model_dump') else {}
			for name, params in data.items():
				if params is None:
					continue
				if isinstance(params, dict):
					if name == 'click':
						label = f'click #{params.get("index", "?")}'
					elif name == 'input':
						text = self._clean_status_text(params.get('text', ''))
						label = f'type "{text[:30]}"'
					elif name == 'scroll':
						label = f'scroll {"down" if params.get("down", True) else "up"}'
					elif name == 'extract':
						label = 'extract page data'
					elif name == 'done':
						label = 'finish'
					else:
						label = name
				else:
					label = name
				labels.append(label)
				break
		if len(action_list) > 3:
			labels.append(f'+{len(action_list) - 3} more')
		return ', '.join(labels)

	def _clean_status_text(self, text: Any, max_length: int = 180) -> str:
		"""Normalize model text for the small visible status bubble."""
		if text is None:
			return ''
		cleaned = ' '.join(str(text).split())
		if len(cleaned) > max_length:
			return cleaned[: max_length - 1] + '...'
		return cleaned

	def _create_tools(self, task: PetTask, session: PetSession) -> Tools:
		"""Create the default browser tools with a pet-specific human input action."""
		# read_canvas/place_pixels are excluded on purpose: they are per-action "Way 1" canvas
		# tools that bounce every pixel batch through the LLM (slow + lossy relay). Canvas work
		# must go through run_helper (ctx.js) instead — one func reads and paints in code.
		tools = Tools(exclude_actions=['evaluate', 'read_canvas', 'place_pixels'])
		tools.action_timeout_overrides['ask_human'] = _PET_STEP_TIMEOUT_SECONDS
		tools.action_timeout_overrides['ask_human_before_click'] = _PET_STEP_TIMEOUT_SECONDS
		tools.action_timeout_overrides['react_to_visible_items'] = _PET_STEP_TIMEOUT_SECONDS
		tools.action_timeout_overrides['run_site_skill'] = _PET_STEP_TIMEOUT_SECONDS

		@tools.action(
			'Record the visible text and identity of a page element by its index. '
			'Call this BEFORE making a decision based on what an element shows '
			'(e.g. which month a calendar is on, what option is selected, what a counter reads). '
			'This records the element in the trace so generated skill code can reuse the exact selector.'
		)
		async def read_element(index: int, label: str, browser_session: BrowserSession) -> ActionResult:
			try:
				node = await browser_session.get_element_by_index(index)
				if node is None:
					return ActionResult(error=f'No element at index {index}')
				text = (node.get_all_children_text(max_depth=2) or '').strip()
				attrs = getattr(node, 'attributes', {}) or {}
				el_id = attrs.get('id', '')
				aria_label = attrs.get('aria-label', '')
				el_class = attrs.get('class', '')
				tag = getattr(node, 'tag_name', '') or ''
				selector = (
					f'#{el_id}'
					if el_id
					else (
						f'[aria-label="{aria_label}"]'
						if aria_label
						else f'{tag}.{el_class.split()[0]}'
						if el_class
						else tag or 'unknown'
					)
				)
				return ActionResult(
					extracted_content=f'{label} ({selector}): "{text}"',
					include_in_memory=True,
				)
			except Exception as e:
				return ActionResult(error=f'read_element failed: {e}')

		@tools.action(
			'Click the same control multiple times in a row, e.g. Previous/Next month arrows, pagination, or Load more. '
			'Re-finds the control between clicks so page re-renders are safe. '
			'Prefer this over queueing the same click action repeatedly.'
		)
		async def click_repeatedly(index: int, count: int, browser_session: BrowserSession) -> ActionResult:
			if count < 1:
				return ActionResult(error='count must be at least 1.')
			if count > 15:
				return ActionResult(error='count must be 15 or less.')
			node = await browser_session.get_element_by_index(index)
			if node is None:
				return ActionResult(error=f'Element index {index} is no longer available. Refresh browser state and retry.')
			identity = self._control_identity(node)
			label = identity.get('aria') or identity.get('ax') or identity.get('id') or f'element #{index}'
			clicked = 0
			for click_number in range(count):
				if session.cancel_event.is_set():
					return ActionResult(error='Task stopped by the user.')
				if click_number > 0:
					selector_map = await self._fresh_selector_map(browser_session)
					node = self._find_control_by_identity(selector_map, identity)
					if node is None:
						return ActionResult(
							error=f'Control {label!r} disappeared after {clicked} of {count} clicks; page may have changed.',
							extracted_content=f'Clicked {label!r} {clicked} of {count} times.',
							include_in_memory=True,
						)
				click_event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
				await click_event
				await click_event.event_result(raise_if_any=True, raise_if_none=False)
				clicked += 1
				await asyncio.sleep(0.4)
			return ActionResult(
				extracted_content=f'Clicked {label!r} {clicked} of {count} times.',
				include_in_memory=True,
			)

		@tools.action(
			'Write and immediately run a small Python helper to perform a fast or repeated sequence on THIS page in '
			'one shot — playing a song on a piano, drawing, or firing many similar clicks/keystrokes. Use this INSTEAD '
			'of repeating the same kind of action many times through normal steps: the helper runs with no further LLM '
			'steps, so it is fast and can keep timing/rhythm. '
			'Your code must define `async def run(ctx):`. Reference page elements ONLY by their integer index from '
			'browser_state, and list every index you use in element_indices — they resolve to the real, verified '
			'elements (a made-up index raises). '
			'ctx API: `await ctx.click(index)`, `await ctx.type(index, text)`, `await ctx.press(keys)` '
			'(a REAL key press like "ArrowUp"/"Enter"/"ctrl+a" — for games/key-driven UIs), '
			'`await ctx.type_text(text, per_char_delay=0.0)` (types a whole string as REAL per-character '
			'keystrokes into the focused element — use this for typing tests / editors / anything that counts '
			'real keypresses, where setting .value is ignored; click the typing area first), `await ctx.wait(seconds)`, '
			'`await ctx.click_at(x, y)` (a REAL click at a viewport coordinate), `await ctx.drag(path)` (a real '
			'click-and-drag stroke through [(x,y), ...]), `await ctx.snapshot()` (per-canvas count+hash+rect), and '
			'`await ctx.js(script)` to run JavaScript and return a value. '
			'To DRAW on a <canvas> (pixels have no index), select the pen, then use ctx.click_at / ctx.drag at the '
			'screen coordinate of each pixel — real OS-level events. '
			'VERIFY by what YOUR action changed, not the absolute page state (the source image is already on screen and '
			'will fool an absolute "is it there" check). Use ctx.snapshot() BEFORE and AFTER a draw and compare: the canvas '
			'whose count/hash CHANGED is the real editable surface; if NOTHING changed, your draw did not land — fix the '
			'input (synthetic ctx.js painting is often ignored; use ctx.click_at/ctx.drag) before continuing. '
			'Use ctx.js freely to READ pixels and compute coordinates. '
			'ACT, do not inspect-and-report: write ONE helper that performs the whole job AND checks its own result '
			'in code, then return only a tiny verdict like "matched: true" or "diff: 137" — never return page data, '
			'pixel arrays, or lists of globals for you to reason over (that floods your context and stalls you). '
			'If you do not know an API, do not loop probing the page and reading results back: prefer the generic '
			'approach that always works — read pixels with getImageData and WRITE by simulating real user input '
			'(dispatch pointer/mouse events on the canvas), all inside the one helper, and self-verify by reading back. '
			'Verify a few elements first (click one or two keys) so you know the index→meaning mapping before authoring.'
		)
		async def run_helper(code: str, element_indices: list[int], browser_session: BrowserSession) -> ActionResult:
			return await self._run_authored_helper(session, code, element_indices, browser_session)

		@tools.action(
			'Run a self-driving game loop INSIDE the page for real-time games (snake, flappy, anything that moves on '
			'its own and needs reactions faster than one-move-at-a-time). Unlike run_helper (whose loop runs out here and '
			'lags one round-trip per move — too slow, you act on stale state and crash), THIS runs your whole loop in the '
			'page at native speed: it reads state and presses keys locally with no lag. '
			'`setup_js` is JavaScript that STARTS a loop (use requestAnimationFrame or setInterval) and must: '
			'(1) keep `window.__PET_AUTOPILOT = {done, score, ticks, death, error}` updated every tick — score is the '
			'current game score (number), ticks counts loop iterations, death is a small object describing HOW it ended '
			'(e.g. {reason:"hit_self", head:[x,y], dir:"left"}); '
			'(2) set `__PET_AUTOPILOT.done = true` when the game ends; '
			'(3) stop its loop as soon as `window.__PET_AUTOPILOT_STOP` is true. '
			'Read game state from JS if reachable (the game object / DOM), else from the canvas via getImageData; press '
			'keys with real in-page KeyboardEvents on document. We run it up to `max_seconds`, then hand you back the '
			'final score and the DEATH state — study the death, then rewrite a smarter loop and call this again. You are '
			'the slow brain that writes and fixes the fast reflex between deaths.'
		)
		async def run_autopilot(setup_js: str, max_seconds: float, browser_session: BrowserSession) -> ActionResult:
			return await self._run_autopilot(session, setup_js, max_seconds, browser_session)

		@tools.action(
			'Run an active site-native skill for this website. Use this for known site UI operations listed in the active skills section. Inputs must match the skill inputs.'
		)
		async def run_site_skill(name: str, inputs: dict[str, Any], browser_session: BrowserSession) -> ActionResult:
			return await self._run_active_skill(task, name, inputs, browser_session)

		@tools.action(
			'Your private notepad for THIS run. Pass a short lesson you just learned '
			'(e.g. "swiping up breaks the corner stack — avoid it"; "the usage number is the bold figure under Billing") '
			'to add it; it returns your full notepad. Consult it before deciding and add to it as you go, so you stop '
			'repeating a mistake within this same run. Keep each note short and concrete.'
		)
		async def note(insight: str) -> ActionResult:
			text = ' '.join(insight.split())
			if text and text not in session.notes:
				session.notes.append(text)
			body = '\n'.join(f'{index}. {entry}' for index, entry in enumerate(session.notes, start=1)) or '(empty)'
			return ActionResult(extracted_content=f'Your notepad for this run:\n{body}', include_in_memory=True)

		@tools.action('Ask the human a question when required information is missing. Wait for their reply before continuing.')
		async def ask_human(question: str) -> ActionResult:
			session.status = PetStatus(state='waiting', message='Waiting for your reply', question=question, task_id=task.id)
			reply_task = asyncio.create_task(session.reply_queue.get())
			cancel_task = asyncio.create_task(session.cancel_event.wait())
			done, pending = await asyncio.wait({reply_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
			for pending_task in pending:
				pending_task.cancel()

			if cancel_task in done and cancel_task.result():
				return ActionResult(error='Task stopped by the user.')

			reply = reply_task.result()
			session.status = PetStatus(state='running', message='Working...', task_id=task.id)
			return ActionResult(extracted_content=f'The human replied: {reply}', include_in_memory=True)

		@tools.action(
			'React to multiple visible items without asking the human. Use only for explicitly requested low-risk reactions like upvote, downvote, like, favorite, save, or follow.'
		)
		async def react_to_visible_items(
			reaction: Literal['upvote', 'downvote', 'like', 'favorite', 'save', 'follow'],
			count: int,
			browser_session: BrowserSession,
		) -> ActionResult:
			if count < 1:
				return ActionResult(error='count must be at least 1.')
			if count > 20:
				return ActionResult(error='count must be 20 or less.')

			session.status = PetStatus(state='running', message=f'Finding {reaction} targets...', task_id=task.id)
			clicked: list[str] = []
			acted_backend_ids: set[int] = set()
			scroll_attempts = 0
			max_scroll_attempts = max(6, count * 4)

			while len(clicked) < count:
				if session.cancel_event.is_set():
					return ActionResult(error='Task stopped by the user.')

				selector_map = await self._fresh_selector_map(browser_session)
				candidate = self._find_reaction_candidate(selector_map, reaction, acted_backend_ids)
				if candidate is None:
					if scroll_attempts >= max_scroll_attempts:
						break
					scroll_attempts += 1
					session.status = PetStatus(
						state='running',
						message=f'Scrolling for more {reaction} targets...',
						task_id=task.id,
					)
					scroll_event = browser_session.event_bus.dispatch(ScrollEvent(direction='down', amount=700, node=None))
					await scroll_event
					await scroll_event.event_result(raise_if_any=True, raise_if_none=False)
					await asyncio.sleep(0.25)
					continue

				index, node = candidate
				session.status = PetStatus(
					state='running',
					message=f'{reaction.title()} {len(clicked) + 1} of {count}...',
					task_id=task.id,
				)
				click_event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
				await click_event
				click_metadata = await click_event.event_result(raise_if_any=True, raise_if_none=False)
				acted_backend_ids.add(node.backend_node_id)
				clicked.append(self._reaction_target_summary(node, index))
				await asyncio.sleep(0.35)

				if isinstance(click_metadata, dict) and click_metadata.get('backend_node_id') is not None:
					acted_backend_ids.add(int(click_metadata['backend_node_id']))

			if len(clicked) < count:
				return ActionResult(
					error=f'Only completed {len(clicked)} of {count} {reaction} actions after scrolling.',
					extracted_content='\n'.join(clicked) if clicked else None,
					include_in_memory=True,
				)

			return ActionResult(
				extracted_content=(
					f'Completed {len(clicked)} {reaction} action(s):\n'
					+ '\n'.join(f'{i}. {target}' for i, target in enumerate(clicked, start=1))
				),
				include_in_memory=True,
			)

		@tools.action(
			'Ask the human to approve a prepared final action. If approved, click the provided final button index immediately.'
		)
		async def ask_human_before_click(question: str, index: int, browser_session: BrowserSession) -> ActionResult:
			session.status = PetStatus(state='waiting', message='Waiting for your approval', question=question, task_id=task.id)
			reply_task = asyncio.create_task(session.reply_queue.get())
			cancel_task = asyncio.create_task(session.cancel_event.wait())
			done, pending = await asyncio.wait({reply_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
			for pending_task in pending:
				pending_task.cancel()

			if cancel_task in done and cancel_task.result():
				return ActionResult(error='Task stopped by the user.')

			reply = reply_task.result().strip().lower()
			approved = reply in {'y', 'yes', 'yep', 'yeah', 'sure', 'ok', 'okay', 'approve', 'approved', 'send', 'submit', 'post'}
			if not approved:
				session.status = PetStatus(state='running', message='Working...', task_id=task.id)
				return ActionResult(
					extracted_content=f'The human did not approve the click. Human replied: {reply}',
					include_in_memory=True,
				)

			node = await browser_session.get_element_by_index(index)
			if node is None:
				session.status = PetStatus(state='running', message='Working...', task_id=task.id)
				return ActionResult(error=f'Element index {index} is no longer available. Refresh browser state and ask again.')

			event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
			await event
			click_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)
			session.status = PetStatus(state='running', message='Working...', task_id=task.id)
			return ActionResult(
				extracted_content='The human approved, and the prepared final button was clicked.',
				include_in_memory=True,
				metadata=click_metadata if isinstance(click_metadata, dict) else None,
			)

		return tools

	async def _fresh_selector_map(self, browser_session: BrowserSession) -> dict[int, EnhancedDOMTreeNode]:
		"""Fetch a fresh visible selector map for deterministic pet tools."""
		state = await browser_session.get_browser_state_summary(include_screenshot=False, cached=False)
		assert state.dom_state is not None
		return state.dom_state.selector_map

	def _control_identity(self, node: EnhancedDOMTreeNode) -> dict[str, str | None]:
		"""Capture stable identity for re-finding a control across page re-renders."""
		attrs = node.attributes or {}
		ax_name = node.ax_node.name if node.ax_node else None
		return {
			'tag': (node.tag_name or '').lower(),
			'aria': attrs.get('aria-label'),
			'id': attrs.get('id'),
			'ax': ax_name,
		}

	def _find_control_by_identity(
		self, selector_map: dict[int, EnhancedDOMTreeNode], identity: dict[str, str | None]
	) -> EnhancedDOMTreeNode | None:
		"""Re-find a control in a fresh selector map by its strongest stable attribute."""
		match_key = next((key for key in ('aria', 'id', 'ax') if identity.get(key)), None)
		if match_key is None:
			return None
		for _, node in sorted(selector_map.items()):
			if (node.tag_name or '').lower() != identity['tag']:
				continue
			if self._control_identity(node).get(match_key) == identity[match_key]:
				return node
		return None

	def _find_reaction_candidate(
		self,
		selector_map: dict[int, EnhancedDOMTreeNode],
		reaction: str,
		acted_backend_ids: set[int],
	) -> tuple[int, EnhancedDOMTreeNode] | None:
		"""Choose the first visible, inactive reaction control from the current selector map."""
		for index, node in sorted(selector_map.items()):
			if node.backend_node_id in acted_backend_ids:
				continue
			if not self._is_visible_node(node):
				continue
			if self._is_probably_ad_node(node):
				continue
			if self._is_nested_source_item(node):
				continue
			if not self._matches_reaction(node, reaction):
				continue
			if self._is_active_reaction(node, reaction):
				continue
			return index, node
		return None

	def _matches_reaction(self, node: EnhancedDOMTreeNode, reaction: str) -> bool:
		"""Return whether a node's accessible text looks like the requested reaction."""
		text = self._node_accessible_text(node).lower()
		if not text:
			return False
		matchers = {
			'upvote': ('upvote',),
			'downvote': ('downvote',),
			'like': ('like',),
			'favorite': ('favorite', 'favourite'),
			'save': ('save', 'bookmark'),
			'follow': ('follow',),
		}
		negative_words = {
			'upvote': ('remove upvote', 'unupvote'),
			'downvote': ('remove downvote', 'undownvote'),
			'like': ('unlike',),
			'favorite': ('unfavorite', 'unfavourite'),
			'save': ('unsave', 'saved', 'remove bookmark'),
			'follow': ('unfollow', 'following'),
		}
		if any(word in text for word in negative_words.get(reaction, ())):
			return False
		return any(word in text for word in matchers.get(reaction, (reaction,)))

	def _is_active_reaction(self, node: EnhancedDOMTreeNode, reaction: str) -> bool:
		"""Detect controls that already represent the requested active reaction."""
		attrs = {key.lower(): str(value).lower() for key, value in node.attributes.items()}
		for key in ('aria-pressed', 'pressed', 'aria-selected', 'selected', 'data-pressed', 'data-state'):
			value = attrs.get(key)
			if value in {'true', 'pressed', 'selected', 'checked', 'on'}:
				return True

		if node.ax_node and node.ax_node.properties:
			for prop in node.ax_node.properties:
				if str(prop.name).lower() in {'pressed', 'selected', 'checked'} and prop.value is True:
					return True

		text = self._node_accessible_text(node).lower()
		active_text = {
			'like': ('unlike',),
			'favorite': ('unfavorite', 'unfavourite'),
			'save': ('unsave', 'saved'),
			'follow': ('unfollow', 'following'),
		}
		return any(word in text for word in active_text.get(reaction, ()))

	def _is_visible_node(self, node: EnhancedDOMTreeNode) -> bool:
		"""Check the visibility flags and viewport-sized rect when available."""
		if node.is_visible is False:
			return False
		rect = node.snapshot_node.clientRects if node.snapshot_node else None
		if rect is not None and (rect.width <= 0 or rect.height <= 0):
			return False
		return True

	def _is_probably_ad_node(self, node: EnhancedDOMTreeNode) -> bool:
		"""Avoid obvious promoted/ad posts for generic low-risk reaction automation."""
		current: EnhancedDOMTreeNode | None = node
		for _ in range(8):
			if current is None:
				return False
			tag = current.tag_name.lower()
			attr_text = ' '.join(str(value).lower() for value in current.attributes.values())
			if tag == 'shreddit-ad-post' or 'promoted' in attr_text or 'advertis' in attr_text:
				return True
			current = current.parent_node
		return False

	def _is_nested_source_item(self, node: EnhancedDOMTreeNode) -> bool:
		"""Skip reactions inside embedded/quoted posts rather than top-level feed items."""
		source_item_depth = 0
		current: EnhancedDOMTreeNode | None = node
		for _ in range(18):
			if current is None:
				return False
			if self._is_source_item_node(current):
				source_item_depth += 1
				if source_item_depth > 1:
					return True
			current = current.parent_node
		return False

	def _is_source_item_node(self, node: EnhancedDOMTreeNode) -> bool:
		"""Return whether a node is a post/tweet/article container."""
		attrs = {key.lower(): str(value).lower() for key, value in node.attributes.items()}
		tag = node.tag_name.lower()
		return tag in {'article', 'shreddit-post'} or attrs.get('role') == 'article' or attrs.get('data-testid') == 'tweet'

	def _node_accessible_text(self, node: EnhancedDOMTreeNode) -> str:
		"""Collect concise text that identifies an interactive node."""
		parts: list[str] = []
		if node.ax_node:
			parts.extend(part for part in [node.ax_node.name, node.ax_node.description, node.ax_node.role] if part)
		for attr in ('aria-label', 'title', 'alt', 'value', 'id', 'name', 'placeholder', 'data-testid'):
			value = node.attributes.get(attr)
			if value:
				parts.append(value)
		child_text = node.get_all_children_text(max_depth=2)
		if child_text:
			parts.append(child_text)
		return ' '.join(str(part) for part in parts if part).strip()

	def _reaction_target_summary(self, node: EnhancedDOMTreeNode, index: int) -> str:
		"""Create a compact summary of a clicked reaction target."""
		current: EnhancedDOMTreeNode | None = node
		best_text = ''
		for _ in range(5):
			if current is None:
				break
			text = current.get_all_children_text(max_depth=2).strip()
			if len(text) > len(best_text):
				best_text = text
			current = current.parent_node
		best_text = ' '.join(best_text.split())
		if len(best_text) > 180:
			best_text = best_text[:177] + '...'
		return best_text or f'element index {index}'

	async def _focus_originating_tab(self, browser: Browser, task: PetTask) -> str:
		"""Focus the tab that submitted the extension task."""
		matching_target = None
		if task.origin_token:
			matching_target = await self._find_target_by_pet_task_marker(browser, task.origin_token)
			if matching_target is None:
				raise RuntimeError(
					'Could not find the exact Chrome page that submitted this task token. '
					'Reload the extension and reload this deployed page, then run the task again.'
				)
		else:
			matching_target = await self._find_target_by_pet_session_marker(browser, task.session_id)
		if matching_target is None:
			raise RuntimeError(
				'Could not find the exact Chrome tab that submitted the task. '
				'Reload the extension and reload this deployed page so the pet session marker is installed.'
			)
		print(
			f'Pet origin target locked: session={task.session_id} '
			f'target={matching_target.target_id[:8]} url={matching_target.url}'
		)

		await self._bind_browser_to_origin_target(browser, matching_target.target_id)
		return matching_target.target_id

	async def _bind_browser_to_origin_target(self, browser: Browser, target_id: str) -> None:
		"""Point browser-use at the exact pet tab without visually activating another Chrome tab."""
		if browser.agent_focus_target_id != target_id:
			print(
				f'Pet focus drift detected; rebinding target {target_id[:8]} '
				f'from {browser.agent_focus_target_id[:8] if browser.agent_focus_target_id else "none"}'
			)
			await browser.get_or_create_cdp_session(target_id=target_id, focus=True)
		# Re-assert anti-throttle every bind (cheap, and a reconnect makes a fresh session that loses it).
		await self._keep_tab_active(browser, target_id)

	async def _keep_tab_active(self, browser: Browser, target_id: str) -> None:
		"""Stop Chrome from throttling the pet's tab when the user switches to another tab/app.

		Backgrounded tabs get their timers/rendering throttled and CDP can go silent (the
		"did not respond within 60s" deaths). Focus emulation makes the page behave as if it is
		always foregrounded, so a long run survives the user using the rest of their browser.
		Best-effort: tolerate Chrome builds that lack a given CDP method.
		"""
		try:
			cdp_session = await browser.get_or_create_cdp_session(target_id=target_id, focus=False)
		except Exception:
			return
		for send in (
			lambda: cdp_session.cdp_client.send.Emulation.setFocusEmulationEnabled(
				params={'enabled': True}, session_id=cdp_session.session_id
			),
			lambda: cdp_session.cdp_client.send.Page.setWebLifecycleState(
				params={'state': 'active'}, session_id=cdp_session.session_id
			),
		):
			try:
				await send()
			except Exception:
				pass

	async def _find_target_by_pet_task_marker(self, browser: Browser, origin_token: str) -> Any | None:
		"""Find the Chrome target whose page DOM has this one-time task marker."""
		return await self._find_target_by_dom_marker(
			browser,
			'data-browser-use-pet-origin-token',
			origin_token,
			'Pet task marker scan did not find requested origin token.',
		)

	async def _find_target_by_pet_session_marker(self, browser: Browser, session_id: str) -> Any | None:
		"""Find the Chrome target whose page DOM has this pet session marker."""
		return await self._find_target_by_dom_marker(
			browser,
			'data-browser-use-pet-session-id',
			session_id,
			'Pet tab marker scan did not find requested session.',
		)

	async def _find_target_by_dom_marker(
		self,
		browser: Browser,
		attribute: str,
		expected_value: str,
		miss_message: str,
	) -> Any | None:
		"""Find the Chrome target whose page DOM has an exact marker attribute."""
		expression = f'document.documentElement.getAttribute({json.dumps(attribute)})'
		seen_markers: list[str] = []
		for target in browser.get_page_targets():
			try:
				cdp_session = await browser.get_or_create_cdp_session(target.target_id, focus=False)
				result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={'expression': expression, 'returnByValue': True},
					session_id=cdp_session.session_id,
				)
			except Exception:
				continue
			marker = result.get('result', {}).get('value')
			if isinstance(marker, str):
				seen_markers.append(f'{target.url} -> {marker}')
			if marker == expected_value:
				return target
		if seen_markers:
			print(f'{miss_message} Saw:\n  ' + '\n  '.join(seen_markers))
		return None

	@staticmethod
	def _json_response(payload: dict[str, Any], status: int = 200) -> web.Response:
		return web.json_response(
			payload,
			status=status,
			headers={
				'Access-Control-Allow-Origin': '*',
				'Access-Control-Allow-Headers': 'Content-Type',
				'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
			},
		)


class PetAccessLogger(AbstractAccessLogger):
	"""Hide frequent status polling while preserving useful bridge request logs."""

	def log(self, request: web.BaseRequest, response: web.StreamResponse, time: float) -> None:
		if request.path == '/status':
			return
		self.logger.info(
			'%s "%s %s HTTP/%s" %s',
			request.remote,
			request.method,
			request.path_qs,
			request.version,
			response.status,
		)


async def serve(host: str, port: int, cdp_url: str | None, max_steps: int, log_dir: Path, site_dir: Path) -> None:
	"""Run the extension bridge until interrupted."""
	bridge = PetBridge(cdp_url=cdp_url, max_steps=max_steps, log_dir=log_dir, site_dir=site_dir)
	app = web.Application()
	app.router.add_get('/health', bridge.health)
	app.router.add_get('/status', bridge.get_status)
	app.router.add_post('/tasks', bridge.submit)
	app.router.add_post('/reply', bridge.reply)
	app.router.add_post('/stop', bridge.stop)
	app.router.add_options('/tasks', bridge.options)
	app.router.add_options('/reply', bridge.options)
	app.router.add_options('/stop', bridge.options)

	runner = web.AppRunner(app, access_log_class=PetAccessLogger)
	await runner.setup()
	site = web.TCPSite(runner, host, port)
	await site.start()
	print(f'Website Pet bridge listening on http://{host}:{port}')
	print('Leave this running. The extension will send tasks from whichever Chrome tab you are viewing.')
	print(f'Detailed per-tab logs: {log_dir}')
	print(f'Per-site pet memory: {site_dir}')
	try:
		await bridge.run_forever()
	finally:
		await runner.cleanup()


def main() -> None:
	"""Parse CLI options and start the extension bridge."""
	parser = argparse.ArgumentParser(description='Run the Website Pet local bridge.')
	parser.add_argument('--host', default='127.0.0.1')
	parser.add_argument('--port', type=int, default=8765)
	parser.add_argument(
		'--cdp-url', default=None, help='Optional Chrome CDP URL. Auto-discovers a debug-enabled Chrome by default.'
	)
	parser.add_argument('--max-steps', type=int, default=30)
	parser.add_argument('--log-dir', type=Path, default=Path('.pet_logs'), help='Directory for per-tab Website Pet logs.')
	parser.add_argument('--site-dir', type=Path, default=Path('.pet_sites'), help='Directory for per-site Website Pet memory.')
	parser.add_argument(
		'--propose-skill',
		default=None,
		metavar='SITE',
		help='Read recent traces for SITE and save a proposal-only skill candidate instead of starting the bridge.',
	)
	parser.add_argument('--proposal-trace-limit', type=int, default=5, help='Number of recent traces to use for --propose-skill.')
	args = parser.parse_args()
	try:
		if args.propose_skill:
			proposal = asyncio.run(
				PetSkillLearner(args.site_dir).propose(
					args.propose_skill,
					_pet_llm(),
					limit=max(1, args.proposal_trace_limit),
				)
			)
			print(proposal.model_dump_json(indent=2))
			return
		asyncio.run(serve(args.host, args.port, args.cdp_url, args.max_steps, args.log_dir, args.site_dir))
	except KeyboardInterrupt:
		print('\nWebsite Pet bridge stopped.')


if __name__ == '__main__':
	main()
