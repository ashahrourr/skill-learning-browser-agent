"""Per-site memory handling for Website Pet."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from browser_use.llm.messages import SystemMessage, UserMessage

MAX_SITE_MEMORY_NOTES = 15
MEMORY_REFLECTION_STEP_THRESHOLD = 8


class SiteMemoryReflection(BaseModel):
	"""Reusable site-navigation notes learned from a completed task."""

	notes: list[str] = Field(default_factory=list, max_length=3)
	# Existing memory notes this run proved wrong (e.g. an "X doesn't exist" note the run
	# disproved by doing X). These get removed so a stale note can't keep misleading runs.
	outdated_notes: list[str] = Field(default_factory=list, max_length=5)


class PetSiteMemory:
	"""Manage per-site memory files and memory reflection."""

	def __init__(self, site_dir: Path):
		self.site_dir = site_dir

	def task_hostname(self, url: str) -> str:
		"""Return the normalized hostname for a submitted site-native task."""
		hostname = urlparse(url).hostname or 'unknown-site'
		return hostname.removeprefix('www.').lower()

	def site_path(self, url: str) -> Path:
		"""Return the local memory folder for the task's current site."""
		site_name = re.sub(r'[^a-zA-Z0-9_.-]+', '-', self.task_hostname(url)).strip('-') or 'unknown-site'
		return self.site_dir / site_name

	def ensure_site_memory(self, url: str) -> Path:
		"""Create a basic site memory file if this deployed site has none yet."""
		site_path = self.site_path(url)
		site_path.mkdir(parents=True, exist_ok=True)
		(site_path / 'traces').mkdir(exist_ok=True)
		(site_path / 'skills').mkdir(exist_ok=True)
		manifest_path = site_path / 'site.json'
		if not manifest_path.exists():
			manifest_path.write_text(
				f'{{\n  "site": "{self.task_hostname(url)}",\n  "version": 1,\n  "skills": []\n}}\n',
				encoding='utf-8',
			)
		memory_path = site_path / 'memory.md'
		if not memory_path.exists():
			memory_path.write_text(
				f'# {self.task_hostname(url)} Pet Memory\n\n'
				'The pet should still use the website UI manually. Use these notes only as navigation hints.\n\n'
				'## Notes\n'
				'- No learned notes yet.\n',
				encoding='utf-8',
			)
		return memory_path

	def prompt(self, url: str, recent_traces: list[dict[str, str]], active_skill_lines: list[str]) -> str:
		"""Load reusable site notes and recent successes for the agent prompt."""
		memory_path = self.ensure_site_memory(url)
		memory = memory_path.read_text(encoding='utf-8').strip()
		if len(memory) > 5000:
			memory = memory[-5000:]

		trace_text = ''
		if recent_traces:
			trace_text = '\n\nRecent successful tasks on this site:\n' + '\n'.join(
				f'- Request: {trace["task"]}\n  Result: {trace["result"]}' for trace in recent_traces
			)

		skill_text = ''
		if active_skill_lines:
			skill_text = '\n\nActive site skills available through run_site_skill:\n' + '\n'.join(active_skill_lines)

		return (
			'\n\nSite memory for this deployed website. '
			'Use memory notes to avoid repeating mistakes and choose better navigation steps. '
			'Active skills are explicit site UI capabilities: call one only when it matches the needed operation, '
			'provide the requested inputs, and then continue from the live website state. '
			'If a skill does not match the visible UI state or required capability, operate manually so the pet can learn. '
			'Skills must still operate the website UI; they must not answer from cached data:\n'
			f'{memory}{trace_text}{skill_text}'
		)

	async def reflect(
		self,
		url: str,
		task_text: str,
		llm: Any,
		final_result: str | None,
		completed_items: list[str],
	) -> SiteMemoryReflection:
		"""Ask the LLM for reusable website lessons."""
		memory_path = self.ensure_site_memory(url)
		current_memory = memory_path.read_text(encoding='utf-8')
		compact_items = '\n'.join(f'- {item}' for item in completed_items[-5:])
		try:
			response = await llm.ainvoke(
				[
					SystemMessage(
						content=(
							'Extract reusable site-navigation memory for a browser pet.\n'
							'Return at most 3 short notes.\n'
							'Only include lessons that would help future manual browsing on the same website.\n'
							'Only include notes learned from friction, retries, confusing UI, validation, slow loading, misleading page state, or non-obvious controls.\n'
							'Good notes mention where to click, which fields reject direct typing, which UI state is misleading, or which page section contains data.\n'
							'Do not add obvious workflow notes like "go to historical results", "extract raw data", "calculate manually", or "the site does not summarize the answer".\n'
							'Do not include final answers, lottery numbers, job titles, user-specific data, generic safety rules, or one-off task results.\n'
							'Do not add obvious notes that are already implied by current memory.\n'
							'\n'
							'Writing about limitations / things that seem missing:\n'
							'- Prefer positive, actionable notes ("the year dropdown is at the top of the date picker — use it to jump years").\n'
							'- Only claim something is unavailable if you actually looked for it this run and it was not there.\n'
							'- Phrase any such limitation as a defeasible observation tied to what you saw '
							'("did not find a year dropdown in the start-date picker this run"), NEVER as an absolute law '
							'("there is no year dropdown, always click the arrow"). Absolute absence claims trap future runs '
							'into never re-checking and become self-reinforcing when they are wrong.\n'
							'\n'
							'Correcting stale memory:\n'
							'- In outdated_notes, copy any EXISTING memory.md note that THIS run proved wrong — especially an '
							'"X does not exist / is not possible" note you disproved by successfully doing X. Those notes will be removed.\n'
							'\n'
							'If nothing reusable was learned and nothing was disproved, return empty lists.'
						)
					),
					UserMessage(
						content=(
							f'Site: {self.task_hostname(url)}\n'
							f'URL: {url}\n'
							f'User request: {task_text}\n\n'
							f'Current memory.md:\n{current_memory[-4000:]}\n\n'
							f'Completed item summaries:\n{compact_items or "(none)"}\n\n'
							f'Final result:\n{(final_result or "")[:3000]}'
						)
					),
				],
				output_format=SiteMemoryReflection,
			)
		except Exception as e:
			print(f'Site memory reflection failed for {self.task_hostname(url)}: {e}')
			return SiteMemoryReflection()

		return response.completion

	def append_notes(self, url: str, notes: list[str]) -> list[str]:
		"""Append non-duplicate memory notes to this site's memory file."""
		if not notes:
			return []
		memory_path = self.ensure_site_memory(url)
		current_memory = memory_path.read_text(encoding='utf-8')
		current_lower = current_memory.lower()
		added: list[str] = []
		for note in notes:
			clean_note = ' '.join(note.split())
			if len(clean_note) < 20 or len(clean_note) > 260:
				continue
			if clean_note.lower() in current_lower:
				continue
			added.append(clean_note)

		if not added:
			return []

		updated_memory = current_memory.replace('- No learned notes yet.\n', '')
		existing_notes = self.site_memory_notes(updated_memory)
		kept_notes = [*existing_notes, *added][-MAX_SITE_MEMORY_NOTES:]
		with memory_path.open('w', encoding='utf-8') as memory_file:
			header = updated_memory.split('## Notes', 1)[0].rstrip()
			memory_file.write(f'{header}\n\n## Notes\n')
			for note in kept_notes:
				memory_file.write(f'- {note}\n')
		return added

	def remove_notes(self, url: str, notes_to_remove: list[str]) -> list[str]:
		"""Remove memory notes a run proved wrong; returns the notes actually removed.

		Matching is fuzzy because the reflection may paraphrase the note it is retiring: an
		existing note is removed when one is a substring of the other or they share most words.
		"""
		if not notes_to_remove:
			return []
		memory_path = self.ensure_site_memory(url)
		current_memory = memory_path.read_text(encoding='utf-8')
		existing = self.site_memory_notes(current_memory)
		if not existing:
			return []
		targets = [' '.join(note.split()).lower() for note in notes_to_remove if note.strip()]
		removed: list[str] = []
		kept: list[str] = []
		for note in existing:
			if self._note_matches_any(note, targets):
				removed.append(note)
			else:
				kept.append(note)
		if not removed:
			return []
		header = current_memory.split('## Notes', 1)[0].rstrip()
		body = ''.join(f'- {note}\n' for note in kept) or '- No learned notes yet.\n'
		memory_path.write_text(f'{header}\n\n## Notes\n{body}', encoding='utf-8')
		return removed

	@staticmethod
	def _note_matches_any(note: str, targets: list[str]) -> bool:
		"""True when `note` is the same lesson as one of the target strings (fuzzy)."""
		normalized = ' '.join(note.split()).lower()
		note_tokens = set(normalized.split())
		for target in targets:
			if not target:
				continue
			if target in normalized or normalized in target:
				return True
			target_tokens = set(target.split())
			if note_tokens and target_tokens:
				jaccard = len(note_tokens & target_tokens) / len(note_tokens | target_tokens)
				if jaccard >= 0.6:
					return True
		return False

	def site_memory_notes(self, memory: str) -> list[str]:
		"""Extract bullet notes from a site memory file."""
		notes: list[str] = []
		for line in memory.splitlines():
			stripped = line.strip()
			if stripped.startswith('- ') and stripped != '- No learned notes yet.':
				notes.append(stripped[2:].strip())
		return notes

	def should_reflect(self, history: Any | None) -> bool:
		"""Return whether a successful run had enough friction to learn a site lesson."""
		if history is None:
			return False
		try:
			if history.has_errors():
				return True
			if history.number_of_steps() >= MEMORY_REFLECTION_STEP_THRESHOLD:
				return True
			action_names = history.action_names()
		except Exception:
			return False
		repeated_actions = {'wait', 'scroll', 'extract', 'search_page', 'find_elements'}
		return any(action_names.count(action) >= 3 for action in repeated_actions)
