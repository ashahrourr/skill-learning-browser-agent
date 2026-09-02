"""Simple operation traces for Website Pet runs."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

TraceOperationKind = Literal[
	'navigation_action',
	'ui_action',
	'data_action',
	'file_action',
	'finish_action',
	'skill_action',
]


class TraceActionContext(BaseModel):
	"""Visible-step context for one or more low-level actions."""

	action_numbers: list[int] = Field(default_factory=list, min_length=1, max_length=30)
	goal: str = ''
	result: str = ''
	outcome: Literal['success', 'failed', 'uncertain'] = 'uncertain'


class TraceOperation(BaseModel):
	"""One compact operation in a saved Website Pet trace."""

	kind: TraceOperationKind = 'ui_action'
	goal: str = Field(default='', max_length=300)
	result: str = Field(default='', max_length=300)
	outcome: Literal['success', 'failed', 'uncertain'] = 'uncertain'
	actions: list[str] = Field(default_factory=list, min_length=1, max_length=30)
	# Element facts recorded at interaction time. The skill learner chooses
	# targets from these — it may not invent selectors the trace cannot prove.
	elements: list[dict[str, Any]] = Field(default_factory=list, max_length=12)


class PetTraceBuilder:
	"""Build compact evidence traces from browser-use history."""

	def final_result_summary(self, final_result: str | None) -> str:
		"""Return a compact final result summary for saved traces."""
		if not final_result:
			return ''
		cleaned = ' '.join(final_result.split())
		attachments_index = cleaned.lower().find('attachments:')
		if attachments_index >= 0:
			cleaned = cleaned[:attachments_index].strip()
		return cleaned[:240]

	def operations(self, task_text: str, history: Any | None, visible_steps: list[str] | None = None) -> list[TraceOperation]:
		"""Return compact operation records for a completed run."""
		action_summaries = self.action_summaries(history, visible_steps or [])
		if not action_summaries:
			return []

		operations: list[TraceOperation] = []
		current_key: tuple[str, str, str] | None = None
		current_actions: list[dict[str, Any]] = []
		for action in action_summaries:
			key = (
				str(action.get('step_goal') or ''),
				str(action.get('step_result') or ''),
				str(action.get('outcome') or 'uncertain'),
			)
			if current_actions and key != current_key:
				operations.append(self.operation_from_actions(task_text, current_key, current_actions))
				current_actions = []
			current_key = key
			current_actions.append(action)
		if current_actions:
			operations.append(self.operation_from_actions(task_text, current_key, current_actions))
		return operations

	def action_summaries(self, history: Any | None, visible_steps: list[str]) -> list[dict[str, Any]]:
		"""Return normalized low-level action facts with visible-step context."""
		if history is None:
			return []
		try:
			model_actions = history.model_actions()
		except Exception:
			return []

		context_by_action = self.trace_action_contexts(visible_steps, len(model_actions))
		aligned_results = self.aligned_action_results(history)
		summaries: list[dict[str, Any]] = []
		for action_number, action in enumerate(model_actions, start=1):
			if not isinstance(action, dict):
				continue
			action_name = next((key for key in action if key != 'interacted_element'), None)
			if not action_name:
				continue
			raw_params = action.get(action_name)
			params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
			context = context_by_action.get(action_number)
			summaries.append(
				{
					'action_number': action_number,
					'action': action_name,
					'target_text': self.interacted_element_text(action.get('interacted_element')),
					'target_css': self.interacted_element_descriptor(action.get('interacted_element')),
					'element': self.element_fingerprint(action.get('interacted_element')),
					'params': self.public_action_params(params),
					'observation': self.action_observation(action_name, aligned_results, action_number),
					'step_goal': context.goal if context else '',
					'step_result': context.result if context else '',
					'outcome': context.outcome if context else 'uncertain',
				}
			)
		return summaries

	def aligned_action_results(self, history: Any) -> list[Any]:
		"""Return per-action results aligned with model_actions() ordering.

		A step's action queue can abort early, leaving fewer results than actions,
		so missing slots are padded with None to keep global indices aligned.
		"""
		aligned: list[Any] = []
		for item in getattr(history, 'history', None) or []:
			model_output = getattr(item, 'model_output', None)
			if model_output is None:
				continue
			actions = list(getattr(model_output, 'action', None) or [])
			results = list(getattr(item, 'result', None) or [])
			for index in range(len(actions)):
				aligned.append(results[index] if index < len(results) else None)
		return aligned

	def action_observation(self, action_name: str, aligned_results: list[Any], action_number: int) -> str:
		"""Return recorded observation text for actions whose result matters to skills."""
		if action_name != 'read_element':
			return ''
		if action_number - 1 >= len(aligned_results):
			return ''
		result = aligned_results[action_number - 1]
		content = getattr(result, 'extracted_content', None) if result is not None else None
		return self.clean_status_text(content, max_length=220) if content else ''

	def trace_action_contexts(self, visible_steps: list[str], max_action_number: int) -> dict[int, TraceActionContext]:
		"""Map visible pet step messages to action numbers and next-step results."""
		step_ranges: list[dict[str, Any]] = []
		next_action_number = 1
		for visible_step in visible_steps:
			action_count = self.visible_step_action_count(visible_step)
			if action_count < 1:
				continue
			start = next_action_number
			end = min(max_action_number, next_action_number + action_count - 1)
			if start > max_action_number:
				break
			step_ranges.append(
				{
					'start': start,
					'end': end,
					'goal': self.visible_step_field(visible_step, 'Now') or '',
					'last': self.visible_step_field(visible_step, 'Last') or '',
				}
			)
			next_action_number = end + 1

		contexts: dict[int, TraceActionContext] = {}
		for index, step_range in enumerate(step_ranges):
			result_range = step_ranges[index + 1] if index + 1 < len(step_ranges) else step_range
			result = self.clean_status_text(result_range['last'], max_length=300)
			context = TraceActionContext(
				action_numbers=list(range(int(step_range['start']), int(step_range['end']) + 1)),
				goal=self.clean_status_text(step_range['goal'], max_length=300),
				result=result,
				outcome=self.outcome_from_text(result),
			)
			for action_number in context.action_numbers:
				contexts[action_number] = context
		return contexts

	def operation_from_actions(
		self, task_text: str, key: tuple[str, str, str] | None, actions: list[dict[str, Any]]
	) -> TraceOperation:
		"""Build one compact operation from consecutive normalized actions."""
		goal, result, outcome = key if key else ('', '', 'uncertain')
		action_labels = [self.action_label(action) for action in actions]
		elements: list[dict[str, Any]] = []
		for action in actions:
			fingerprint = action.get('element')
			if fingerprint and fingerprint not in elements and len(elements) < 12:
				elements.append(fingerprint)
		return TraceOperation(
			kind=self.operation_kind(actions),
			goal=self.clean_status_text(goal, max_length=300),
			result=self.clean_status_text(result, max_length=300),
			outcome=outcome if outcome in {'success', 'failed', 'uncertain'} else 'uncertain',  # type: ignore[arg-type]
			actions=self.compress_actions(action_labels),
			elements=elements,
		)

	def operation_kind(self, actions: list[dict[str, Any]]) -> TraceOperationKind:
		"""Classify one operation from action names only."""
		action_names = [str(action.get('action') or '') for action in actions]
		if any(action_name == 'done' for action_name in action_names):
			return 'finish_action'
		if any(action_name == 'run_site_skill' for action_name in action_names):
			return 'skill_action'
		if any(action_name in {'write_file', 'read_file'} for action_name in action_names):
			return 'file_action'
		if any(action_name in {'extract', 'search_page'} for action_name in action_names):
			return 'data_action'
		if any(action_name in {'navigate', 'go_back'} for action_name in action_names):
			return 'navigation_action'
		return 'ui_action'

	def action_label(self, action: dict[str, Any]) -> str:
		"""Return a compact human-readable action label."""
		action_name = str(action.get('action') or 'action')
		target = action.get('target_text')
		raw_params = action.get('params')
		params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
		if action_name == 'click':
			if target:
				return f'click {target}'
			# Bracketed index form so an unlabeled element can never be mistaken
			# for a real DOM id by the skill learner downstream; the tag.class
			# descriptor gives the learner a targetable selector for anonymous elements.
			descriptor = action.get('target_css')
			index_label = f'[index {params.get("index", "?")}]'
			return f'click {index_label} ({descriptor})' if descriptor else f'click {index_label}'
		if action_name == 'input':
			text = self.clean_status_text(params.get('text') or '', max_length=80)
			return f'type {text!r} into {target or "field"}'
		if action_name == 'select_dropdown':
			text = self.clean_status_text(params.get('text') or '', max_length=80)
			return f'select {text!r} in {target or "dropdown"}'
		if action_name == 'scroll':
			return f'scroll {"down" if params.get("down", True) else "up"}'
		if action_name == 'wait':
			return f'wait {params.get("seconds", 1)}s'
		if action_name == 'read_element':
			observation = action.get('observation')
			if observation:
				return f'read {observation}'
			label = params.get('label') or 'element'
			return f'read {label}'
		if action_name == 'click_repeatedly':
			return f'click {target or "control"} x{params.get("count", "?")}'
		if action_name == 'run_site_skill':
			return 'run site skill'
		if action_name == 'search_page':
			query = params.get('query') or params.get('text')
			if query:
				return f'search page for {self.clean_status_text(query, max_length=80)!r}'
			return 'search page'
		if action_name == 'extract':
			return 'extract page data'
		if action_name == 'write_file':
			return 'write results file'
		if action_name == 'read_file':
			return 'read results file'
		if action_name == 'done':
			return 'finish task'
		return action_name

	def compress_actions(self, actions: list[str]) -> list[str]:
		"""Merge repeated adjacent actions into compact labels."""
		compressed: list[str] = []
		index = 0
		while index < len(actions):
			action = actions[index]
			run_end = index + 1
			while run_end < len(actions) and actions[run_end] == action:
				run_end += 1
			count = run_end - index
			compressed.append(f'{action} x{count}' if count >= 2 else action)
			index = run_end
		return compressed[:30]

	def visible_step_field(self, visible_step: str, field_name: str) -> str | None:
		"""Extract a named field from the visible pet step text."""
		next_fields = {
			'Last': ('Now', 'ActionCount', 'Action'),
			'Now': ('ActionCount', 'Action'),
			'ActionCount': ('Action',),
			'Action': (),
		}
		start_match = re.search(rf'(?:^|\n){re.escape(field_name)}:\s*', visible_step)
		if not start_match:
			return None
		start = start_match.end()
		end = len(visible_step)
		for next_field in next_fields.get(field_name, ()):
			next_match = re.search(rf'(?:^|\n){re.escape(next_field)}:\s*', visible_step[start:])
			if next_match:
				end = start + next_match.start()
				break
		value = visible_step[start:end].strip()
		return value or None

	def visible_step_action_count(self, visible_step: str) -> int:
		"""Return the number of low-level actions a visible pet step represents."""
		count_text = self.visible_step_field(visible_step, 'ActionCount')
		if count_text and count_text.strip().isdigit():
			return int(count_text.strip())
		# Fallback for older traces without ActionCount field
		action_text = self.visible_step_field(visible_step, 'Action')
		if not action_text:
			return 0
		more_match = re.search(r'\+(\d+)\s+more', action_text)
		more_count = int(more_match.group(1)) if more_match else 0
		action_text = action_text[: more_match.start()] if more_match else action_text
		parts = [part.strip() for part in action_text.split(',') if part.strip()]
		return len(parts) + more_count

	def interacted_element_text(self, element: Any) -> str | None:
		"""Return stable visible text or label from an interacted element snapshot."""
		if element is None:
			return None
		if isinstance(element, dict):
			ax_name = element.get('ax_name')
			attributes = element.get('attributes') or {}
			node_value = element.get('node_value')
		else:
			ax_name = getattr(element, 'ax_name', None)
			attributes = getattr(element, 'attributes', None) or {}
			node_value = getattr(element, 'node_value', None)
		for value in (
			ax_name,
			attributes.get('aria-label'),
			attributes.get('placeholder'),
			attributes.get('id'),
			attributes.get('name'),
			attributes.get('title'),
			attributes.get('data-testid'),
			node_value,
		):
			if value:
				return self.clean_status_text(value, max_length=200)
		return None

	def element_fingerprint(self, element: Any) -> dict[str, Any] | None:
		"""Facts about an element captured at interaction time, empty fields dropped.

		These are the only values the skill learner may build targets from; the
		x_path is kept so identical twins (same tag+classes) stay distinguishable.
		"""
		if element is None:
			return None
		if isinstance(element, dict):
			attributes = element.get('attributes') or {}
			node_name = element.get('node_name')
			ax_name = element.get('ax_name')
			node_value = element.get('node_value')
			x_path = element.get('x_path')
		else:
			attributes = getattr(element, 'attributes', None) or {}
			node_name = getattr(element, 'node_name', None)
			ax_name = getattr(element, 'ax_name', None)
			node_value = getattr(element, 'node_value', None)
			x_path = getattr(element, 'x_path', None)
		if not node_name:
			return None
		label = next(
			(
				value
				for value in (
					attributes.get('placeholder'),
					attributes.get('name'),
					attributes.get('title'),
					attributes.get('data-testid'),
				)
				if value
			),
			None,
		)
		fingerprint = {
			'tag': str(node_name).lower(),
			'id': attributes.get('id'),
			'classes': str(attributes.get('class') or '').split()[:8],
			'aria': ax_name or attributes.get('aria-label'),
			'label': label,
			'text': self.clean_status_text(node_value, max_length=80) or None,
			'x_path': self.clean_status_text(x_path, max_length=160) or None,
		}
		return {key: value for key, value in fingerprint.items() if value}

	def interacted_element_descriptor(self, element: Any) -> str | None:
		"""Return a 'tag.class' targeting hint for elements with no accessible name."""
		if element is None:
			return None
		if isinstance(element, dict):
			node_name = element.get('node_name')
			attributes = element.get('attributes') or {}
		else:
			node_name = getattr(element, 'node_name', None)
			attributes = getattr(element, 'attributes', None) or {}
		if not node_name:
			return None
		tag = str(node_name).lower()
		class_tokens = str(attributes.get('class') or '').split()
		return f'{tag}.{class_tokens[0]}' if class_tokens else tag

	def public_action_params(self, params: dict[str, Any]) -> dict[str, Any]:
		"""Keep only non-sensitive action fields needed for compact labels."""
		allowed = {'text', 'query', 'down', 'pages', 'seconds', 'index', 'count', 'label'}
		return {key: value for key, value in params.items() if key in allowed}

	def outcome_from_text(self, text: str) -> Literal['success', 'failed', 'uncertain']:
		"""Classify a visible step result conservatively."""
		lower = text.lower()
		if any(word in lower for word in ('failed', 'failure', 'not successfully', "didn't", 'did not', 'error')):
			return 'failed'
		if any(word in lower for word in ('success', 'successfully', 'worked', 'verified', 'confirmed', 'ready')):
			return 'success'
		return 'uncertain'

	def clean_status_text(self, text: Any, max_length: int = 180) -> str:
		"""Normalize model text for compact trace fields."""
		if text is None:
			return ''
		cleaned = ' '.join(str(text).split())
		if len(cleaned) > max_length:
			return cleaned[: max_length - 3] + '...'
		return cleaned
