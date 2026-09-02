"""Browser interaction context passed to site skill code."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger('browser_use.pet_skill_context')

from browser_use import BrowserSession
from browser_use.browser.events import ClickElementEvent, SelectDropdownOptionEvent, TypeTextEvent
from browser_use.dom.views import EnhancedDOMTreeNode

# Shadow-DOM-piercing query helpers injected into every script. Plain
# document.querySelector cannot see inside shadow roots. Note: JS can only
# pierce OPEN shadow roots — closed roots are reachable only via the CDP
# DOM tree, which is why interactions prefer node dispatch over JS.
_DEEP_QUERY_JS = (
	'const __deepAll = (selector) => {'
	'  const out = [];'
	'  const walk = (root) => {'
	'    if (!root || !root.querySelectorAll) return;'
	'    for (const el of root.querySelectorAll(selector)) out.push(el);'
	'    for (const el of root.querySelectorAll("*")) { if (el.shadowRoot) walk(el.shadowRoot); }'
	'  };'
	'  walk(document);'
	'  return out;'
	'};'
	'const __deepOne = (selector) => __deepAll(selector)[0] ?? null;'
	# Twin controls (e.g. start/end date pickers) share selectors; only the OPEN one is
	# visible, so visibility-filtering resolves to the right twin without guessing state classes.
	'const __isVisible = (el) => {'
	'  if (!el) return false;'
	'  const s = window.getComputedStyle(el);'
	'  if (s.display === "none" || s.visibility === "hidden" || parseFloat(s.opacity || "1") === 0) return false;'
	# offsetParent is null when an ancestor is display:none (the closed twin picker), while still
	# allowing zero-size controls like CSS-triangle prev/next arrows that a width/height check would drop.
	'  if (el.offsetParent === null && s.position !== "fixed") return false;'
	'  return true;'
	'};'
	'const __deepVisible = (selector) => __deepAll(selector).filter(__isVisible);'
	'const __deepOneVisible = (selector) => __deepVisible(selector)[0] ?? null;'
	'const __fireToggles = (el) => {'
	'  el.dispatchEvent(new Event("change", {bubbles:true}));'
	'  el.dispatchEvent(new Event("input", {bubbles:true}));'
	'};'
)


def _normalize_label(value: str) -> str:
	"""Normalize an accessible name for tolerant comparison."""
	return ' '.join(value.split()).casefold()


def _node_identity_names(node: 'EnhancedDOMTreeNode') -> list[str]:
	"""All names an element is known by — mirrors the trace labeler's fields."""
	attrs = node.attributes or {}
	names: list[str] = []
	if node.ax_node and node.ax_node.name:
		names.append(str(node.ax_node.name))
	for attr in ('aria-label', 'placeholder', 'title', 'name', 'id'):
		value = attrs.get(attr)
		if value:
			names.append(str(value))
	return names


# Simple selector forms we can resolve against the CDP DOM tree:
# '#id', 'tag#id', '.class', 'tag.class', 'tag.class1.class2', '[aria-label="x"]', 'tag'
_SIMPLE_SELECTOR_RE = re.compile(
	r'^(?P<tag>[a-zA-Z][a-zA-Z0-9-]*)?(?:#(?P<id>[\w-]+)|(?P<cls>(?:\.[\w-]+)+)|\[aria-label=["\'](?P<aria>[^"\']+)["\']\])?$'
)


class SkillContext:
	"""High-level browser helpers for site skill entrypoints.

	Passed as `ctx` to every skill's run(inputs, ctx) function.
	Interactions (click/type/select) resolve elements through the browser's
	CDP DOM tree — the same path the agent uses — so they work inside open
	AND closed shadow roots. JS-based deep query is the fallback for complex
	CSS selectors and read operations.
	All methods raise on failure — let exceptions propagate so the skill
	registry returns a clean ActionResult(error=...).
	"""

	def __init__(self, browser_session: BrowserSession):
		self._session = browser_session

	# ------------------------------------------------------------------
	# Element resolution via the CDP DOM tree
	# ------------------------------------------------------------------

	async def _find_node(self, predicate: Callable[[EnhancedDOMTreeNode], bool], nth: int = 1) -> EnhancedDOMTreeNode | None:
		"""Return the nth interactive node (1-based) matching the predicate from a fresh DOM snapshot."""
		assert nth >= 1, 'nth is 1-based'
		state = await self._session.get_browser_state_summary(include_screenshot=False, cached=False)
		if state.dom_state is None:
			return None
		seen = 0
		for _, node in sorted(state.dom_state.selector_map.items()):
			try:
				if predicate(node):
					seen += 1
					if seen == nth:
						return node
			except Exception:
				continue
		return None

	def _simple_selector_predicate(self, selector: str) -> Callable[[EnhancedDOMTreeNode], bool] | None:
		"""Build a node predicate for simple selectors; None means use the JS fallback."""
		match = _SIMPLE_SELECTOR_RE.match(selector.strip())
		if not match or not any(match.group(g) for g in ('tag', 'id', 'cls', 'aria')):
			return None
		tag, el_id, cls, aria = match.group('tag'), match.group('id'), match.group('cls'), match.group('aria')

		def predicate(node: EnhancedDOMTreeNode) -> bool:
			attrs = node.attributes or {}
			if tag and (node.tag_name or '').lower() != tag.lower():
				return False
			if el_id and attrs.get('id') != el_id:
				return False
			if cls:
				node_classes = (attrs.get('class') or '').split()
				if any(wanted not in node_classes for wanted in cls.lstrip('.').split('.')):
					return False
			if aria:
				ax_name = (node.ax_node.name or '').strip() if node.ax_node else ''
				if attrs.get('aria-label') != aria and ax_name != aria:
					return False
			return True

		return predicate

	async def _dispatch_click(self, node: EnhancedDOMTreeNode) -> None:
		event = self._session.event_bus.dispatch(ClickElementEvent(node=node))
		await event
		await event.event_result(raise_if_any=True, raise_if_none=False)

	# ------------------------------------------------------------------
	# Low-level
	# ------------------------------------------------------------------

	async def js(self, script: str) -> Any:
		"""Execute arbitrary JavaScript and return the result value."""
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

	async def _deep_js(self, body: str) -> Any:
		"""Run a script body with shadow-DOM-piercing query helpers in scope."""
		return await self.js(f'(function(){{ {_DEEP_QUERY_JS} {body} }})()')

	async def wait(self, seconds: float = 1.0) -> None:
		"""Wait for a number of seconds."""
		await asyncio.sleep(seconds)

	# ------------------------------------------------------------------
	# Navigation
	# ------------------------------------------------------------------

	async def navigate(self, url: str) -> None:
		"""Navigate to a URL and wait for the page to settle."""
		cdp_session = await self._session.get_or_create_cdp_session(focus=True)
		await cdp_session.cdp_client.send.Page.navigate(
			params={'url': url},
			session_id=cdp_session.session_id,
		)
		await asyncio.sleep(1.5)

	async def current_url(self) -> str:
		"""Return the current page URL."""
		return await self.js('window.location.href') or ''

	# ------------------------------------------------------------------
	# Clicking
	# ------------------------------------------------------------------

	async def click(self, selector: str, nth: int = 1) -> None:
		"""Click element by CSS selector. Simple selectors use the CDP DOM tree (works in closed shadow roots).

		nth picks the nth match in document order (1-based) when several identical
		elements share the selector, e.g. twin calendar-icon spans.
		"""
		predicate = self._simple_selector_predicate(selector)
		if predicate is not None:
			node = await self._find_node(predicate, nth=nth)
			if node is not None:
				logger.info(f'🔧 skill click {selector!r} nth={nth} → DOM node <{node.tag_name}> "{(node.get_all_children_text(max_depth=1) or "").strip()[:40]}"')
				await self._dispatch_click(node)
				return
		logger.info(f'🔧 skill click {selector!r} nth={nth} → JS deep query (not in interactive map)')
		# Complex selector or not in the interactive map — try JS deep query.
		# Prefer visible matches so a shared selector hits the open twin, not a hidden one.
		await self._deep_js(
			f'const matches = __deepVisible({json.dumps(selector)});'
			f'const el = (matches.length ? matches : __deepAll({json.dumps(selector)}))[{nth - 1}] ?? null;'
			f'if (!el) throw new Error("No match {nth} for: " + {json.dumps(selector)});'
			f'el.click();'
			f'__fireToggles(el);'
		)

	async def click_by_text(self, text: str, tag: str = '*') -> None:
		"""Click the first element whose visible text exactly matches."""
		target = text.strip()

		def predicate(node: EnhancedDOMTreeNode) -> bool:
			if tag != '*' and (node.tag_name or '').lower() != tag.lower():
				return False
			ax_name = (node.ax_node.name or '').strip() if node.ax_node else ''
			if ax_name == target:
				return True
			return node.get_all_children_text(max_depth=2).strip() == target

		node = await self._find_node(predicate)
		if node is not None:
			await self._dispatch_click(node)
			return
		await self._deep_js(
			f'const target = {json.dumps(target)};'
			f'for (const el of __deepAll({json.dumps(tag)})) {{'
			f'  if ((el.textContent || "").trim() === target) {{ el.click(); __fireToggles(el); return; }}'
			f'}}'
			f'throw new Error("No element with text: " + target);'
		)

	async def click_by_aria_label(self, label: str, partial: bool = True) -> None:
		"""Click element by aria-label / accessible name.

		partial=True clicks the first containing match. partial=False uses a match
		ladder: exact -> normalized (case/whitespace) -> unique substring match,
		because real accessible names often differ from traces in trivial ways.
		"""

		def node_label(node: EnhancedDOMTreeNode) -> str | None:
			value = (node.attributes or {}).get('aria-label')
			if value is None and node.ax_node:
				value = node.ax_node.name
			return value

		if partial:

			def predicate(node: EnhancedDOMTreeNode) -> bool:
				value = node_label(node)
				return value is not None and label.lower() in value.lower()

			node = await self._find_node(predicate)
			if node is not None:
				logger.info(f'🔧 skill click_by_aria_label {label!r} partial → DOM node <{node.tag_name}>')
				await self._dispatch_click(node)
				return
			logger.info(f'🔧 skill click_by_aria_label {label!r} partial → JS deep query')
			await self._deep_js(
				f'const needle = {json.dumps(label.lower())};'
				f'const match = (list) => list.find(e => e.getAttribute("aria-label").toLowerCase().includes(needle));'
				f'const el = match(__deepVisible("[aria-label]")) ?? match(__deepAll("[aria-label]"));'
				f'if (!el) throw new Error("No aria-label containing: " + {json.dumps(label)});'
				f'el.click();'
				f'__fireToggles(el);'
			)
			return

		# Exact mode matches the same identity fields the trace labeler records
		# (ax name, aria-label, placeholder, title, name, id), so any name a
		# trace shows for an element is resolvable here.
		normalized_target = _normalize_label(label)
		state = await self._session.get_browser_state_summary(include_screenshot=False, cached=False)
		labeled: list[tuple[EnhancedDOMTreeNode, list[str]]] = []
		if state.dom_state is not None:
			for _, candidate in sorted(state.dom_state.selector_map.items()):
				names = _node_identity_names(candidate)
				if names:
					labeled.append((candidate, names))

		exact = [n for n, names in labeled if label in names]
		if not exact:
			exact = [n for n, names in labeled if any(_normalize_label(v) == normalized_target for v in names)]
		if not exact:
			substring = [n for n, names in labeled if any(normalized_target in _normalize_label(v) for v in names)]
			if len(substring) == 1:
				exact = substring
		if exact:
			logger.info(f'🔧 skill click_by_aria_label {label!r} exact → DOM node <{exact[0].tag_name}> ({len(exact)} match(es))')
			await self._dispatch_click(exact[0])
			return
		logger.info(f'🔧 skill click_by_aria_label {label!r} exact → JS deep query (no map match)')

		await self._deep_js(
			f'const q = {json.dumps(label)};'
			f'const norm = s => s.trim().replace(/\\s+/g, " ").toLowerCase();'
			f'const attrs = ["aria-label", "placeholder", "title", "name", "id"];'
			f'const els = __deepVisible("[aria-label],[placeholder],[title],[name],[id]");'
			f'const namesOf = e => attrs.map(a => e.getAttribute(a)).filter(Boolean);'
			f'let el = els.find(e => namesOf(e).includes(q));'
			f'if (!el) el = els.find(e => namesOf(e).some(v => norm(v) === norm(q)));'
			f'if (!el) {{'
			f'  const subs = els.filter(e => namesOf(e).some(v => norm(v).includes(norm(q))));'
			f'  if (subs.length === 1) el = subs[0];'
			f'}}'
			f'if (!el) {{'
			f'  const near = els.flatMap(e => namesOf(e))'
			f'    .filter(v => norm(v).includes(norm(q).split(" ")[0])).slice(0, 8);'
			f'  throw new Error("No element named: " + q + (near.length ? ". Nearby names: " + near.join(" | ") : ""));'
			f'}}'
			f'el.click();'
			f'__fireToggles(el);'
		)

	# ------------------------------------------------------------------
	# Typing / selecting
	# ------------------------------------------------------------------

	async def type_into(self, selector: str, text: str, clear: bool = True) -> None:
		"""Type text into an input or textarea matched by CSS selector."""
		predicate = self._simple_selector_predicate(selector)
		if predicate is not None:
			node = await self._find_node(predicate)
			if node is not None:
				event = self._session.event_bus.dispatch(TypeTextEvent(node=node, text=text, clear=clear))
				await event
				await event.event_result(raise_if_any=True, raise_if_none=False)
				return
		clear_js = 'el.value = ""; el.dispatchEvent(new Event("input", {bubbles:true}));' if clear else ''
		await self._deep_js(
			f'const el = __deepOne({json.dumps(selector)});'
			f'if (!el) throw new Error("No element: " + {json.dumps(selector)});'
			f'el.focus();'
			f'{clear_js}'
			f'el.value = {json.dumps(text)};'
			f'__fireToggles(el);'
		)

	async def select_option(self, selector: str, value: str) -> None:
		"""Select an option in a <select> by value or visible text."""
		predicate = self._simple_selector_predicate(selector)
		if predicate is not None:
			node = await self._find_node(predicate)
			if node is not None:
				logger.info(f'🔧 skill select_option {selector!r} = {value!r} → DOM node <{node.tag_name}>')
				event = self._session.event_bus.dispatch(SelectDropdownOptionEvent(node=node, text=value))
				await event
				await event.event_result(raise_if_any=True, raise_if_none=False)
				return
		logger.info(f'🔧 skill select_option {selector!r} = {value!r} → JS deep query')
		await self._deep_js(
			f'const el = __deepOneVisible({json.dumps(selector)}) ?? __deepOne({json.dumps(selector)});'
			f'if (!el) throw new Error("No element: " + {json.dumps(selector)});'
			f'const opts = Array.from(el.options || []);'
			f'const opt = opts.find(o => o.value === {json.dumps(value)})'
			f'  || opts.find(o => (o.textContent || "").trim() === {json.dumps(value)});'
			f'if (!opt) throw new Error("No option: " + {json.dumps(value)});'
			f'el.value = opt.value;'
			f'__fireToggles(el);'
		)

	# ------------------------------------------------------------------
	# Waiting / querying
	# ------------------------------------------------------------------

	async def wait_for(self, selector: str, timeout: float = 5.0) -> None:
		"""Wait until a CSS selector matches an element."""
		deadline = asyncio.get_event_loop().time() + timeout
		predicate = self._simple_selector_predicate(selector)
		while asyncio.get_event_loop().time() < deadline:
			found = await self._deep_js(f'return !!__deepOne({json.dumps(selector)});')
			if not found and predicate is not None:
				found = await self._find_node(predicate) is not None
			if found:
				return
			await asyncio.sleep(0.25)
		raise TimeoutError(f'Element not found within {timeout}s: {selector}')

	async def aria_label_exists(self, label: str) -> bool:
		"""Return whether an element with this exact aria-label currently exists."""

		def predicate(node: EnhancedDOMTreeNode) -> bool:
			value = (node.attributes or {}).get('aria-label')
			if value is None and node.ax_node:
				value = node.ax_node.name
			return value == label

		if await self._find_node(predicate) is not None:
			return True
		found = await self._deep_js(
			f'return __deepVisible("[aria-label]").some(e => e.getAttribute("aria-label") === {json.dumps(label)});'
		)
		return bool(found)

	async def get_text(self, selector: str) -> str | None:
		"""Return trimmed text content of first matching element."""
		text = await self._deep_js(f'return __deepOne({json.dumps(selector)})?.textContent?.trim() ?? null;')
		if text is not None:
			return text
		predicate = self._simple_selector_predicate(selector)
		if predicate is not None:
			node = await self._find_node(predicate)
			if node is not None:
				return node.get_all_children_text(max_depth=2).strip()
		return None

	async def get_attribute(self, selector: str, attr: str) -> str | None:
		"""Return an attribute value from first matching element."""
		value = await self._deep_js(f'return __deepOne({json.dumps(selector)})?.getAttribute({json.dumps(attr)}) ?? null;')
		if value is not None:
			return value
		predicate = self._simple_selector_predicate(selector)
		if predicate is not None:
			node = await self._find_node(predicate)
			if node is not None:
				return (node.attributes or {}).get(attr)
		return None
