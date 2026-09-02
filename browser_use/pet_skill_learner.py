"""Proposal-only site skill learning from Website Pet traces."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.pet_skill_steps import SkillSteps, validate_generated_steps


class SiteSkillProposal(BaseModel):
	"""A reviewed-before-writing proposal for a future code-backed site skill."""

	worth_building: bool = False
	name: str = Field(default='', max_length=120)
	description: str = Field(default='', max_length=500)
	inputs: list[str] = Field(default_factory=list, max_length=12)
	trigger: str = Field(default='', max_length=800)
	evidence: list[str] = Field(default_factory=list, max_length=8)
	suggested_steps: list[str] = Field(default_factory=list, max_length=20)
	safety_notes: list[str] = Field(default_factory=list, max_length=8)


class SavedSkillProposal(BaseModel):
	"""A persisted proposal with trace metadata."""

	created_at: str
	site: str
	trace_count: int
	proposal: SiteSkillProposal


class PetSkillLearner:
	"""Read clean traces and ask the LLM for a skill proposal."""

	def __init__(self, site_dir: Path):
		self.site_dir = site_dir

	def site_path(self, site: str) -> Path:
		"""Return the local folder for a site name."""
		site_name = re.sub(r'[^a-zA-Z0-9_.-]+', '-', site.removeprefix('www.').lower()).strip('-')
		return self.site_dir / (site_name or 'unknown-site')

	def load_recent_traces(self, site: str, limit: int = 5) -> list[dict[str, Any]]:
		"""Load the newest successful v2 traces for a site."""
		trace_dir = self.site_path(site) / 'traces'
		if not trace_dir.exists():
			return []
		traces: list[dict[str, Any]] = []
		for trace_path in sorted(trace_dir.glob('*.json'), reverse=True):
			try:
				trace = json.loads(trace_path.read_text(encoding='utf-8'))
			except Exception:
				continue
			if trace.get('version') != 2 or trace.get('successful') is not True:
				continue
			operations = trace.get('operations')
			if not isinstance(operations, list) or not operations:
				continue
			traces.append(
				{
					'task': trace.get('task', ''),
					'url': trace.get('url', ''),
					'final_result_summary': trace.get('final_result_summary', ''),
					'operations': operations[:30],
				}
			)
			if len(traces) >= limit:
				break
		return traces

	def _fit_traces(self, traces: list[dict[str, Any]], char_limit: int = 18000) -> list[dict[str, Any]]:
		"""Return as many traces as fit within char_limit, dropping from the end."""
		kept: list[dict[str, Any]] = []
		total = 0
		for trace in traces:
			chunk = json.dumps(trace, ensure_ascii=False)
			if total + len(chunk) > char_limit:
				break
			kept.append(trace)
			total += len(chunk)
		return kept

	async def propose(
		self, site: str, llm: Any, limit: int = 5, existing_skills: list[str] | None = None
	) -> SavedSkillProposal:
		"""Return and save a proposal for a repeated site workflow.

		`existing_skills` are compact "name(inputs): description" lines for skills this site
		already has. The proposer reconciles against them so it doesn't duplicate one under a
		new name.
		"""
		traces = self.load_recent_traces(site, limit=limit)
		if not traces:
			return SavedSkillProposal(
				created_at=datetime.now(UTC).isoformat(),
				site=site,
				trace_count=0,
				proposal=SiteSkillProposal(
					worth_building=False,
					safety_notes=['No successful v2 traces with operations were found for this site.'],
				),
			)

		fitted_traces = self._fit_traces(traces)

		response = await llm.ainvoke(
			[
				SystemMessage(
					content=(
						'You propose Website Pet site skills from clean operation traces.\n'
						'Return one proposal only.\n'
						'\n'
						'Trace field meanings:\n'
						'- operations: ordered list of high-level steps taken during the run\n'
						'- kind: navigation_action | ui_action | data_action | file_action | finish_action | skill_action\n'
						'- goal: what the agent was trying to do at that step\n'
						'- result: what actually happened\n'
						'- outcome: success | failed | uncertain\n'
						'- actions: low-level browser actions taken (click X, type "Y", scroll down)\n'
						'\n'
						'A good skill is a repeated, site-specific UI workflow that would save future browsing work.\n'
						'Do not propose final-answer calculation, reporting, file writing, or cached-answer skills.\n'
						'The skill must operate the live website UI.\n'
						'Prefer workflows that appear across multiple traces with successful outcomes.\n'
						'If no reusable workflow is worth building, set worth_building=false and leave name empty.\n'
						'\n'
						'Existing skills for this site are listed below. Reconcile against them to avoid duplicates:\n'
						'- If this run\'s workflow is already covered by an existing skill, set worth_building=false.\n'
						'- If it is the same workflow as an existing skill but better/more complete, reuse that '
						"existing skill's EXACT name (it will be updated in place, not duplicated).\n"
						'- Only choose a new name when the workflow is genuinely different from every existing skill.\n'
						'\n'
						'For suggested_steps: write plain English steps a developer can follow to implement the skill,\n'
						'e.g. "1. Navigate to the Jobs page. 2. Click the Experience Level filter button. 3. Select the option matching {experience_level}. 4. Click Show Results."\n'
						'Use {input_name} placeholders for runtime values.\n'
						'Use snake_case for name. Inputs should be runtime values like start_date, end_date, query, option.'
					)
				),
				UserMessage(
					content=(
						f'Site: {site}\n'
						f'Existing skills for this site:\n'
						f'{chr(10).join(existing_skills) if existing_skills else "(none yet)"}\n\n'
						f'Recent successful v2 traces ({len(fitted_traces)} of {len(traces)}):\n'
						f'{json.dumps(fitted_traces, indent=2, ensure_ascii=False)}'
					)
				),
			],
			output_format=SiteSkillProposal,
		)
		saved = SavedSkillProposal(
			created_at=datetime.now(UTC).isoformat(),
			site=site,
			trace_count=len(traces),
			proposal=response.completion,
		)
		self.save_proposal(saved)
		return saved

	@staticmethod
	def _trace_elements(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""All element fingerprints recorded across the traces' operations."""
		elements: list[dict[str, Any]] = []
		for trace in traces:
			for operation in trace.get('operations') or []:
				for element in operation.get('elements') or []:
					if isinstance(element, dict) and element not in elements:
						elements.append(element)
		return elements

	async def generate_skill_steps(self, proposal: SavedSkillProposal, llm: Any) -> Path:
		"""Ask the LLM to fill the declarative step schema for an approved proposal."""
		p = proposal.proposal
		inputs_text = ', '.join(p.inputs) if p.inputs else 'none'
		traces = self._fit_traces(self.load_recent_traces(proposal.site), char_limit=8000)
		traces_text = json.dumps(traces, indent=2, ensure_ascii=False) if traces else 'No traces available.'
		trace_elements = self._trace_elements(traces)

		messages = [
			SystemMessage(
				content=(
					'You convert successful browser operation traces into a declarative site skill: '
					'a list of steps executed by a fixed interpreter. You fill a schema — you do not write code.\n'
					'\n'
					'Step kinds:\n'
					'- navigate: {"do": "navigate", "url": "https://..."}\n'
					'- click:    {"do": "click", "target": {<one of: id, aria, text, css>}}\n'
					'- type:     {"do": "type", "target": {...}, "value": "..."}\n'
					'- select:   {"do": "select", "target": {...}, "value": "..."}  # picks a <select> option by value or visible text\n'
					'- wait:     {"do": "wait", "seconds": 1.5}\n'
					'- probe_until: {"do": "probe_until", "until_aria_exists": "...", "repeat_click": {<target>}, "max_repeats": 12}\n'
					'  Clicks repeat_click until an element with that exact aria-label exists. Use this to page through\n'
					'  calendars/lists when the number of clicks depends on runtime state.\n'
					'\n'
					'Targets — set exactly ONE locator field (plus optional nth):\n'
					'- id:   real DOM id from the trace (e.g. "drawsStartDate"), without "#" — NEVER a bare number\n'
					'- aria: EXACT aria-label or accessible name from the trace\n'
					'- text: EXACT visible text from the trace\n'
					'- css:  only simple selectors ("#id", ".class", "tag.class1.class2", \'[aria-label="x"]\') —\n'
					'  no combinators. All classes must come from ONE recorded element; prefer its most\n'
					'  distinctive class (e.g. "span.icon_pastStart" over the generic "span.fi").\n'
					'- nth:  optional, 1-based match in document order — disambiguates identical elements,\n'
					'  e.g. {"css": "span.icon-calendar", "nth": 2} for the second of two twin calendar icons\n'
					'\n'
					'Placeholders in url/value/aria/text/id/until_aria_exists strings:\n'
					'- {input_name} inserts the runtime input value as-is\n'
					'- {input_name|%m/%d/%Y} parses the input as a date (any common format) and reformats it.\n'
					'  Example: a calendar day cell labeled "04/01/2024" for input start_date is {start_date|%m/%d/%Y}.\n'
					'  Example: a year dropdown value is {start_date|%Y}.\n'
					'  Placeholders may ONLY use the declared skill inputs — adapt these examples to the actual\n'
					'  input names listed below; never copy an example placeholder the skill does not declare.\n'
					'\n'
					'Rules:\n'
					'- Each operation\'s "elements" list holds the facts recorded about every element at the moment\n'
					'  it was interacted with: tag, id, classes, aria (accessible name), label, text, x_path.\n'
					'  CHOOSE every target from these recorded facts — id from "id", aria from "aria", text from\n'
					'  "text", css from tag+classes. Targets not present in any recorded element are REJECTED.\n'
					'- NEVER invent a name from goal/result sentences — goals describe intent ("the Start Date\n'
					'  field"), not the element\'s real accessible name, and invented names fail at runtime.\n'
					'- Two recorded elements with the same tag+classes but different x_path are distinct twins\n'
					'  (e.g. start vs end date icons): target them with css tag.class plus "nth": 1 / "nth": 2\n'
					'  in the order their clicks appear in the trace.\n'
					'- Action labels like "click [index 178]" or "click #20151" with a plain number are browser-use\n'
					'  internal element indices, NOT ids — they change on every page load.\n'
					'- Never depend on when the trace ran: month/year navigation must use probe_until, never a fixed\n'
					'  number of clicks copied from the trace.\n'
					'- For trace actions like "select \'2025\' in Select a year", use a select step with the accessible\n'
					'  name: {"do": "select", "target": {"aria": "Select a year"}, "value": "{start_date|%Y}"}.\n'
					'- Keep the skill minimal: only the steps needed to perform the operation.'
				)
			),
			UserMessage(
				content=(
					f'Skill name: {p.name}\n'
					f'Description: {p.description}\n'
					f'Inputs: {inputs_text}\n'
					f'\nRecent successful traces — derive the steps from these, do not assume anything not shown here:\n'
					f'{traces_text}'
				)
			),
		]
		response = await llm.ainvoke(messages, output_format=SkillSteps)
		skill_steps: SkillSteps = response.completion

		# Prompt rules alone don't stop the LLM from inventing selectors or
		# undeclared placeholders — validate against the declared inputs and the
		# recorded element fingerprints, with one corrective retry before
		# refusing to save anything.
		problems = validate_generated_steps(skill_steps, p.inputs, trace_elements=trace_elements)
		if problems:
			problems_text = '\n'.join(f'- {problem}' for problem in problems)
			retry_messages = [
				*messages,
				UserMessage(
					content=(
						f'Your previous attempt was rejected by the validator:\n{problems_text}\n'
						f'\nPrevious attempt:\n{skill_steps.model_dump_json(indent=2, exclude_none=True)}\n'
						f'\nReturn a corrected skill that fixes every problem. Remember: numeric ids are internal\n'
						f'indices (use aria/text targets instead), and placeholders may only use: {inputs_text}.'
					)
				),
			]
			response = await llm.ainvoke(retry_messages, output_format=SkillSteps)
			skill_steps = response.completion
			problems = validate_generated_steps(skill_steps, p.inputs, trace_elements=trace_elements)
			if problems:
				raise ValueError(f'Generated skill failed validation after retry: {"; ".join(problems)}')

		site_path = self.site_path(proposal.site)
		skills_dir = site_path / 'skills'
		skills_dir.mkdir(parents=True, exist_ok=True)
		safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', p.name.strip().lower()).strip('_')
		metadata = {
			'name': p.name,
			'status': 'trial',
			'version': 1,
			'description': p.description,
			'inputs': p.inputs,
			'trigger': p.trigger,
			'entrypoint': 'steps',
		}
		(skills_dir / f'{safe_name}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
		steps_path = skills_dir / f'{safe_name}.steps.json'
		steps_path.write_text(skill_steps.model_dump_json(indent=2, exclude_none=True), encoding='utf-8')
		self._register_in_manifest(site_path, p.name)
		return steps_path

	async def generate_skill_code(self, proposal: SavedSkillProposal, llm: Any) -> Path:
		"""Ask the LLM to implement the approved proposal as a Python skill file."""
		p = proposal.proposal
		inputs_text = ', '.join(p.inputs) if p.inputs else 'none'
		traces = self._fit_traces(self.load_recent_traces(proposal.site), char_limit=8000)
		traces_text = json.dumps(traces, indent=2, ensure_ascii=False) if traces else 'No traces available.'

		response = await llm.ainvoke(
			[
				SystemMessage(
					content=(
						'You write Python async skill files for a browser automation system.\n'
						'The skill file must define one async function:\n'
						'\n'
						'    async def run(inputs: dict, ctx: SkillContext) -> ActionResult:\n'
						'\n'
						'SkillContext API (import not needed — ctx is injected):\n'
						'  await ctx.navigate(url)                          # go to URL\n'
						'  await ctx.current_url() -> str                   # get current URL\n'
						'  await ctx.wait(seconds)                          # sleep\n'
						'  await ctx.wait_for(css_selector, timeout=5.0)   # wait until element exists\n'
						'  await ctx.click(css_selector)                    # click by CSS selector\n'
						'  await ctx.click_by_text(text, tag="*")          # click by visible text\n'
						'  await ctx.click_by_aria_label(label, partial=True)  # click by aria-label (partial match)\n'
						'  await ctx.click_by_aria_label(label, partial=False) # click by aria-label (exact match) — use for date cells like "05/01/2026"\n'
						'  await ctx.select_option(css_selector, value)   # pick a <select> option by value or visible text\n'
						'  await ctx.aria_label_exists(label) -> bool      # whether an exact aria-label exists right now\n'
						'  await ctx.type_into(css_selector, text)         # type into input\n'
						'  await ctx.get_text(css_selector) -> str|None    # get element text\n'
						'  await ctx.get_attribute(css_selector, attr) -> str|None\n'
						'  await ctx.js(script) -> Any                     # run arbitrary JS\n'
						'\n'
						'ActionResult import:\n'
						'  from browser_use import ActionResult\n'
						'\n'
						'Rules:\n'
						'- Study the traces carefully. Use the EXACT element IDs and names that worked in the traces.\n'
						'  If a trace clicked "drawsStartDate", use ctx.click("#drawsStartDate").\n'
						'  If a trace clicked "Previous month", use ctx.click_by_aria_label("Previous month").\n'
						'  Do NOT invent fallback selectors or try multiple alternatives — use what the traces prove works.\n'
						'- For trace actions like "select \'2025\' in Select a year", use ctx.select_option with the accessible\n'
						'  name as the selector: ctx.select_option(\'[aria-label="Select a year"]\', "2025").\n'
						'  Never click <option> elements directly — that does not work.\n'
						'- To navigate paginated widgets (calendars, result pages) when the trace gives no header selector,\n'
						'  probe with ctx.aria_label_exists(target) and click Previous/Next until it returns True, then click the target.\n'
						"- Never bake in values that depend on WHEN the trace ran (today's date, the month or year a calendar\n"
						'  opened on). Read the current state from the page at runtime instead, using a selector the trace proves exists.\n'
						'- If the trace gives no CSS selector for an element, click it by the visible text or aria-label shown\n'
						'  in the trace (click_by_text / click_by_aria_label). Never invent CSS ids or class names.\n'
						'- Use only SIMPLE selectors: "#id", ".class", "tag", \'[aria-label="x"]\'. Combinators like\n'
						'  "#a + span" or "div > button" do NOT work reliably — use click_by_aria_label or click_by_text instead.\n'
						'- TWINS: when a page has duplicate controls (e.g. separate Start and End date pickers, each with its\n'
						'  own "Previous month"/"Select a year"/date cells), a bare selector hits the wrong one. The ctx click/\n'
						'  select/aria helpers already resolve to the VISIBLE instance — so open exactly one picker at a time and\n'
						'  finish it before opening the other. In any custom ctx.js, filter to visible elements (getBoundingClientRect\n'
						'  width/height > 0 and not display:none) so you read the open picker, never the hidden twin.\n'
						'- For trace actions like "read calendar_header (.some-selector)", use ctx.get_text(".some-selector")\n'
						"  to read the same element in skill code. The value shown in the next operation's result tells you\n"
						'  what the element contained, so you can compare against it for navigation logic.\n'
						'- IMPORTANT: action labels like "click [index 178]", "click #20151" or "click #8116" where the\n'
						'  target is a plain number are browser-use internal index IDs, NOT real CSS selectors. Never use them as CSS selectors.\n'
						'  If the trace only has a numeric index for an element, find it by its relationship to a known element\n'
						'  (e.g. the sibling span of #drawsStartDate) or by aria-label, not by that number.\n'
						'- Input values arrive as strings in unpredictable formats (e.g. dates as "2024-04-01",\n'
						'  "04/01/2024", or "April 1 2024"). Parse defensively — try multiple formats — and normalize\n'
						'  before use. Only return an error if the value is truly unparseable.\n'
						'- Return ActionResult(extracted_content="done: ...") on success.\n'
						'- Return ActionResult(error="...") on failure — never raise.\n'
						'- Use try/except around each major step.\n'
						'- Always wait for page load after navigation or filter application.\n'
						'- Output ONLY the Python code — no markdown fences, no explanation.\n'
					)
				),
				UserMessage(
					content=(
						f'Skill name: {p.name}\n'
						f'Description: {p.description}\n'
						f'Inputs: {inputs_text}\n'
						f'\nRecent successful traces — derive the full implementation from these, '
						f'do not assume anything not shown here:\n{traces_text}'
					)
				),
			]
		)
		raw_completion = getattr(response, 'completion', None) or getattr(response, 'content', None) or response
		code = str(raw_completion).strip()
		# Strip markdown fences if the LLM added them anyway
		if code.startswith('```'):
			code = re.sub(r'^```[a-z]*\n?', '', code)
			code = re.sub(r'\n?```$', '', code)

		# Write metadata JSON + code file
		site_path = self.site_path(proposal.site)
		skills_dir = site_path / 'skills'
		skills_dir.mkdir(parents=True, exist_ok=True)
		safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', p.name.strip().lower()).strip('_')
		metadata = {
			'name': p.name,
			'status': 'trial',
			'version': 1,
			'description': p.description,
			'inputs': p.inputs,
			'trigger': p.trigger,
			'entrypoint': 'run',
		}
		(skills_dir / f'{safe_name}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
		code_path = skills_dir / f'{safe_name}.py'
		code_path.write_text(code, encoding='utf-8')
		try:
			import asyncio

			format_proc = await asyncio.create_subprocess_exec(
				'uv',
				'run',
				'ruff',
				'format',
				str(code_path),
				stdout=asyncio.subprocess.DEVNULL,
				stderr=asyncio.subprocess.DEVNULL,
			)
			await format_proc.wait()
		except Exception:
			pass
		self._register_in_manifest(site_path, p.name)
		return code_path

	def _register_in_manifest(self, site_path: Path, skill_name: str) -> None:
		"""Add or update the skill entry in site.json."""
		manifest_path = site_path / 'site.json'
		site_name = site_path.name
		if manifest_path.exists():
			try:
				data = json.loads(manifest_path.read_text(encoding='utf-8'))
			except Exception:
				data = {'site': site_name, 'version': 1, 'skills': []}
		else:
			data = {'site': site_name, 'version': 1, 'skills': []}

		skills = data.setdefault('skills', [])
		# Update existing entry or append new one
		for entry in skills:
			if entry.get('name') == skill_name:
				entry['status'] = 'trial'
				break
		else:
			skills.append({'name': skill_name, 'status': 'trial', 'version': 1})

		manifest_path.write_text(json.dumps(data, indent=2), encoding='utf-8')

	def save_proposal(self, proposal: SavedSkillProposal) -> Path:
		"""Persist a proposal without activating or creating a skill."""
		proposals_dir = self.site_path(proposal.site) / 'skill_proposals'
		proposals_dir.mkdir(parents=True, exist_ok=True)
		timestamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
		safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', proposal.proposal.name.strip().lower()).strip('_')
		file_stem = safe_name or 'no_skill'
		proposal_path = proposals_dir / f'{timestamp}-{file_stem}.json'
		proposal_path.write_text(proposal.model_dump_json(indent=2), encoding='utf-8')
		return proposal_path
