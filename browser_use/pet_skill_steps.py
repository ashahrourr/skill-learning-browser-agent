"""Declarative step-based site skills: schema models and interpreter.

A skill is a validated list of steps, not freeform code. The LLM fills this
schema from traces; one interpreter (this file) executes every skill through
SkillContext. Mistakes like invented selector syntax, strict date parsing, or
hardcoded calendar months are unrepresentable here.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from browser_use import ActionResult
from browser_use.pet_skill_context import SkillContext

_DATE_FORMATS = ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y', '%d/%m/%Y', '%B %d %Y', '%B %d, %Y', '%b %d %Y', '%b %d, %Y']
_PLACEHOLDER_RE = re.compile(r'\{([a-zA-Z_][\w]*)(?:\|([^}]+))?\}')


class SkillStepTarget(BaseModel):
	"""How to find one element. Exactly one field should be set."""

	model_config = ConfigDict(extra='forbid')

	id: str | None = None  # element id, without '#'
	aria: str | None = None  # exact aria-label / accessible name
	text: str | None = None  # exact visible text
	css: str | None = None  # simple CSS selector: '#id', '.class', 'tag.class1.class2', '[aria-label="x"]'
	nth: int | None = Field(default=None, ge=1, le=20)  # 1-based match in document order, for identical elements


class SkillStep(BaseModel):
	"""One step of a site skill."""

	model_config = ConfigDict(extra='forbid')

	do: Literal['navigate', 'click', 'type', 'select', 'wait', 'probe_until']
	target: SkillStepTarget | None = None
	url: str | None = None  # navigate only
	value: str | None = None  # type/select only
	seconds: float | None = Field(default=None, ge=0, le=30)  # wait only
	# probe_until: click `repeat_click` until an element with aria-label
	# `until_aria_exists` appears (checked before every click).
	until_aria_exists: str | None = None
	repeat_click: SkillStepTarget | None = None
	max_repeats: int = Field(default=12, ge=1, le=40)


class SkillSteps(BaseModel):
	"""A complete declarative site skill."""

	model_config = ConfigDict(extra='forbid')

	steps: list[SkillStep] = Field(min_length=1, max_length=40)


def render_template(template: str, inputs: dict) -> str:
	"""Substitute {input} and {input|strftime} placeholders with input values.

	Date-formatted placeholders parse the input value defensively across common
	formats, so callers may pass '2024-04-01', '04/01/2024', or 'April 1 2024'.
	"""

	def replace(match: re.Match[str]) -> str:
		name, fmt = match.group(1), match.group(2)
		if name not in inputs:
			raise KeyError(f'Skill input {name!r} was not provided')
		value = str(inputs[name]).strip()
		if not fmt:
			return value
		parsed = _parse_date(value)
		if parsed is None:
			raise ValueError(f'Skill input {name!r}={value!r} is not a recognizable date')
		return parsed.strftime(fmt)

	return _PLACEHOLDER_RE.sub(replace, template)


def _parse_date(value: str) -> datetime | None:
	for fmt in _DATE_FORMATS:
		try:
			return datetime.strptime(value, fmt)
		except ValueError:
			continue
	return None


def _step_template_strings(step: SkillStep) -> list[str]:
	"""All strings in a step that may contain {placeholder} templates."""
	strings = [step.url, step.value, step.until_aria_exists]
	for target in (step.target, step.repeat_click):
		if target is not None:
			strings.extend([target.id, target.aria, target.text, target.css])
	return [s for s in strings if s]


class _TraceElementPool:
	"""Identity facts recorded in trace element fingerprints, indexed for provenance checks."""

	def __init__(self, trace_elements: list[dict]):
		self.elements = trace_elements
		self.ids: set[str] = set()
		self.tags: set[str] = set()
		# Names an element answers to in SkillContext's exact-match ladder:
		# ax/aria name, placeholder/name/title label, visible text, id.
		self.identity: set[str] = set()
		for element in trace_elements:
			if element.get('id'):
				self.ids.add(str(element['id']))
			if element.get('tag'):
				self.tags.add(str(element['tag']).lower())
			for field in ('aria', 'label', 'text', 'id'):
				if element.get(field):
					self.identity.add(str(element[field]))

	def _class_combo_recorded(self, tag: str | None, classes: list[str]) -> bool:
		"""True if one recorded element carries the tag AND all the classes together."""
		for element in self.elements:
			if tag and str(element.get('tag') or '').lower() != tag.lower():
				continue
			if set(classes) <= {str(cls) for cls in element.get('classes') or []}:
				return True
		return False

	def css_ok(self, css: str) -> bool:
		css = css.strip()
		if match := re.fullmatch(r'#([\w-]+)', css):
			return match.group(1) in self.ids
		if match := re.fullmatch(r'([a-zA-Z][\w-]*)?((?:\.[\w-]+)+)', css):
			tag, classes = match.group(1), match.group(2).lstrip('.').split('.')
			return self._class_combo_recorded(tag, classes)
		if match := re.fullmatch(r'\[aria-label=["\']([^"\']+)["\']\]', css):
			return match.group(1) in self.identity
		if re.fullmatch(r'[a-zA-Z][\w-]*', css):
			return css.lower() in self.tags
		return False


def validate_generated_steps(
	skill: SkillSteps, declared_inputs: list[str], trace_elements: list[dict] | None = None
) -> list[str]:
	"""Return problems that make an LLM-generated skill unrunnable; empty list means valid.

	Catches the failure modes prompt rules alone cannot prevent: browser-use
	internal element indices stored as DOM ids (e.g. '178' from a trace label
	like 'click #178' — indices change every snapshot and '#178' is not even
	valid CSS), placeholders referencing inputs the skill never declared, and —
	when traces carry element fingerprints — targets the LLM invented rather
	than chose from a recorded element (e.g. aria 'Start Date' when no element
	ever answered to that name). Templated values are exempt from provenance
	since their concrete form only exists at runtime.
	"""
	problems: list[str] = []
	allowed = set(declared_inputs)
	pool = _TraceElementPool(trace_elements) if trace_elements else None
	for step_number, step in enumerate(skill.steps, start=1):
		for role, target in (('target', step.target), ('repeat_click', step.repeat_click)):
			if target is None:
				continue
			if target.id and target.id.isdigit():
				problems.append(
					f'step {step_number}: {role} id {target.id!r} is a browser-use internal element index, '
					f'not a real DOM id — locate this element by its aria-label or text from the trace instead'
				)
			if target.css and re.fullmatch(r'#\d+', target.css.strip()):
				problems.append(
					f'step {step_number}: {role} css {target.css!r} is a browser-use internal element index, '
					f'not a valid CSS selector — locate this element by its aria-label or text from the trace instead'
				)
			if pool is not None:
				problems.extend(
					f'step {step_number}: {role} {field} {value!r} does not appear in any recorded trace element — '
					f'choose targets only from the element facts listed in the traces'
					for field, value, ok in (
						('id', target.id, target.id in pool.ids if target.id else True),
						('aria', target.aria, target.aria in pool.identity if target.aria else True),
						('text', target.text, target.text in pool.identity if target.text else True),
						('css', target.css, pool.css_ok(target.css) if target.css else True),
					)
					if value and '{' not in value and not ok
				)
		for template in _step_template_strings(step):
			for match in _PLACEHOLDER_RE.finditer(template):
				name = match.group(1)
				if name not in allowed:
					problems.append(
						f'step {step_number}: placeholder {{{name}}} is not a declared skill input '
						f'(declared inputs: {sorted(allowed)})'
					)
	return list(dict.fromkeys(problems))


async def run_skill_steps(skill: SkillSteps, inputs: dict, ctx: SkillContext) -> ActionResult:
	"""Execute a declarative skill. Returns ActionResult; never raises."""
	for step_number, step in enumerate(skill.steps, start=1):
		try:
			await _run_step(step, inputs, ctx)
		except Exception as e:
			return ActionResult(error=f'Skill step {step_number} ({step.do}) failed: {e}')
	return ActionResult(extracted_content='done: skill completed all steps', include_in_memory=True)


async def _run_step(step: SkillStep, inputs: dict, ctx: SkillContext) -> None:
	if step.do == 'navigate':
		if not step.url:
			raise ValueError('navigate step requires url')
		await ctx.navigate(render_template(step.url, inputs))
		return

	if step.do == 'wait':
		await ctx.wait(step.seconds if step.seconds is not None else 1.0)
		return

	if step.do == 'click':
		if step.target is None:
			raise ValueError('click step requires target')
		await _click_target(step.target, inputs, ctx)
		return

	if step.do == 'type':
		if step.target is None or step.value is None:
			raise ValueError('type step requires target and value')
		await ctx.type_into(_target_selector(step.target, inputs), render_template(step.value, inputs))
		return

	if step.do == 'select':
		if step.target is None or step.value is None:
			raise ValueError('select step requires target and value')
		await ctx.select_option(_target_selector(step.target, inputs), render_template(step.value, inputs))
		return

	if step.do == 'probe_until':
		if not step.until_aria_exists or step.repeat_click is None:
			raise ValueError('probe_until step requires until_aria_exists and repeat_click')
		needle = render_template(step.until_aria_exists, inputs)
		for _ in range(step.max_repeats):
			if await ctx.aria_label_exists(needle):
				return
			await _click_target(step.repeat_click, inputs, ctx)
			await ctx.wait(0.3)
		if await ctx.aria_label_exists(needle):
			return
		raise RuntimeError(f'Element with aria-label {needle!r} did not appear after {step.max_repeats} repeats')

	raise ValueError(f'Unknown step kind: {step.do}')


async def _click_target(target: SkillStepTarget, inputs: dict, ctx: SkillContext) -> None:
	nth = target.nth or 1
	if target.id:
		await ctx.click(f'#{render_template(target.id, inputs)}', nth=nth)
		return
	if target.aria:
		await ctx.click_by_aria_label(render_template(target.aria, inputs), partial=False)
		return
	if target.text:
		await ctx.click_by_text(render_template(target.text, inputs))
		return
	if target.css:
		await ctx.click(render_template(target.css, inputs), nth=nth)
		return
	raise ValueError('Target has no id, aria, text, or css')


def _target_selector(target: SkillStepTarget, inputs: dict) -> str:
	"""Return a CSS selector for type/select targets."""
	if target.id:
		return f'#{render_template(target.id, inputs)}'
	if target.css:
		return render_template(target.css, inputs)
	if target.aria:
		return f'[aria-label="{render_template(target.aria, inputs)}"]'
	raise ValueError('type/select target needs id, css, or aria')
