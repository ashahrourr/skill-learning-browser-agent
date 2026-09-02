"""Site skill loading and execution for Website Pet."""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, Field

from browser_use import ActionResult, BrowserSession
from browser_use.pet_skill_context import SkillContext
from browser_use.pet_skill_steps import SkillSteps, run_skill_steps

logger = logging.getLogger('browser_use.pet_skills')

TRIAL_USES = 3
TRIAL_MAX_FAILURES = 2


class SiteSkillManifestEntry(BaseModel):
	"""One skill entry in a site's manifest."""

	name: str = Field(min_length=1, max_length=120)
	status: Literal['trial', 'stable', 'inactive'] = 'trial'
	version: int = Field(default=1, ge=1)


class SiteSkillManifest(BaseModel):
	"""Machine-readable index for one deployed site's pet brain."""

	site: str = Field(min_length=1, max_length=240)
	version: int = Field(default=1, ge=1)
	skills: list[SiteSkillManifestEntry] = Field(default_factory=list)


class SiteSkillMetadata(BaseModel):
	"""Metadata describing a code-backed site skill."""

	name: str = Field(min_length=1, max_length=120)
	status: Literal['trial', 'stable', 'inactive'] = 'trial'
	version: int = Field(default=1, ge=1)
	description: str = Field(default='', max_length=500)
	inputs: list[str] = Field(default_factory=list, max_length=20)
	trigger: str = Field(default='', max_length=800)
	capabilities: dict[str, Any] = Field(default_factory=dict)
	entrypoint: str = 'run'
	# Trial tracking
	trial_uses: int = Field(default=0, ge=0)
	trial_failures: int = Field(default=0, ge=0)


class LoadedSiteSkill(BaseModel):
	"""A validated active skill plus its source paths.

	Step-based skills have steps_path set; legacy/escape-hatch Python skills
	have code_path set. steps_path wins when both exist.
	"""

	metadata: SiteSkillMetadata
	metadata_path: Path
	code_path: Path | None = None
	steps_path: Path | None = None
	# A saved live-helper: its verified ctx.js code, replayed by PetBridge (not this registry),
	# because it needs the helper runtime (ctx.click_at/drag/snapshot) + index->selector resolution.
	helper_path: Path | None = None

	model_config = {'arbitrary_types_allowed': True}


class PetSkillRegistry:
	"""Load and execute code-backed skills from a per-site brain folder."""

	def ensure_manifest(self, site_path: Path, site_name: str) -> Path:
		"""Create a default site manifest and skills folder if missing."""
		site_path.mkdir(parents=True, exist_ok=True)
		(site_path / 'skills').mkdir(exist_ok=True)
		manifest_path = site_path / 'site.json'
		if not manifest_path.exists():
			manifest = SiteSkillManifest(site=site_name)
			manifest_path.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')
		return manifest_path

	def load_active_skills(self, site_path: Path) -> list[LoadedSiteSkill]:
		"""Load trial and stable skills listed in site.json."""
		manifest_path = site_path / 'site.json'
		if not manifest_path.exists():
			return []
		try:
			manifest = SiteSkillManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
		except Exception as e:
			print(f'Could not read site skill manifest {manifest_path}: {e}')
			return []

		loaded: list[LoadedSiteSkill] = []
		for entry in manifest.skills:
			if entry.status == 'inactive':
				continue
			safe_name = self.safe_skill_name(entry.name)
			if not safe_name:
				continue
			metadata_path = site_path / 'skills' / f'{safe_name}.json'
			steps_path = site_path / 'skills' / f'{safe_name}.steps.json'
			code_path = site_path / 'skills' / f'{safe_name}.py'
			helper_path = site_path / 'skills' / f'{safe_name}.helper.py'
			if not metadata_path.exists() or not (steps_path.exists() or code_path.exists() or helper_path.exists()):
				continue
			try:
				metadata = SiteSkillMetadata.model_validate_json(metadata_path.read_text(encoding='utf-8'))
			except Exception as e:
				print(f'Could not read site skill metadata {metadata_path}: {e}')
				continue
			if metadata.status == 'inactive':
				continue
			loaded.append(
				LoadedSiteSkill(
					metadata=metadata,
					metadata_path=metadata_path,
					steps_path=steps_path if steps_path.exists() else None,
					code_path=code_path if code_path.exists() else None,
					helper_path=helper_path if helper_path.exists() else None,
				)
			)
		return loaded

	def prompt_lines(self, skills: list[LoadedSiteSkill]) -> list[str]:
		"""Return compact active skill descriptions for the agent prompt."""
		lines: list[str] = []
		for skill in skills:
			metadata = skill.metadata
			line = f'- {metadata.name}({", ".join(metadata.inputs)}): {metadata.description}'
			if metadata.trigger:
				line += f' Trigger: {metadata.trigger}'
			capability_text = self.capability_text(metadata.capabilities)
			if capability_text:
				line += f' Capabilities: {capability_text}'
			lines.append(line)
		return lines

	async def execute(
		self,
		site_path: Path,
		name: str,
		inputs: dict[str, Any],
		browser_session: BrowserSession,
	) -> ActionResult:
		"""Run a skill and update its trial counters."""
		skill = next((s for s in self.load_active_skills(site_path) if s.metadata.name == name), None)
		if skill is None:
			return ActionResult(error=f'No active site skill named {name!r}.')

		missing_inputs = [i for i in skill.metadata.inputs if i not in inputs]
		if missing_inputs:
			return ActionResult(error=f'Site skill {name!r} missing input(s): {", ".join(missing_inputs)}.')

		logger.info(f'🧩 running site skill {name!r} with inputs {inputs}')
		succeeded = False
		hard_failure = False
		try:
			ctx = SkillContext(browser_session)
			if skill.steps_path is not None:
				steps = SkillSteps.model_validate_json(skill.steps_path.read_text(encoding='utf-8'))
				result = await run_skill_steps(steps, inputs, ctx)
			else:
				assert skill.code_path is not None
				module = self.load_module(skill)
				entrypoint = getattr(module, skill.metadata.entrypoint)
				result = entrypoint(inputs=inputs, ctx=ctx)
				if inspect.isawaitable(result):
					result = await result
			succeeded = not (isinstance(result, ActionResult) and result.error)
		except Exception as e:
			# The skill crashed or failed to load — that's a hard failure (disable immediately).
			result = ActionResult(error=f'Site skill {name!r} failed: {e}')
			hard_failure = True

		if succeeded:
			logger.info(f'🧩 site skill {name!r} succeeded')
		else:
			error_text = result.error if isinstance(result, ActionResult) else 'unknown'
			logger.warning(f'🧩 site skill {name!r} failed: {error_text}')

		self._record_trial_outcome(skill, succeeded, hard_failure=hard_failure)

		if isinstance(result, ActionResult):
			return result
		if isinstance(result, dict):
			return ActionResult(extracted_content=json.dumps(result, ensure_ascii=False), include_in_memory=True)
		return ActionResult(extracted_content=str(result), include_in_memory=True)

	def _record_trial_outcome(self, skill: LoadedSiteSkill, succeeded: bool, hard_failure: bool = False) -> None:
		"""Update counters and promote/demote the skill.

		`trial_failures` is a *consecutive* failure streak (reset on success) tracked for both
		trial AND stable skills, so a once-stable skill that breaks (e.g. the site changed) gets
		disabled too instead of failing forever. A `hard_failure` (the skill crashed / failed to
		load) disables it immediately rather than waiting for the streak.
		"""
		metadata = skill.metadata
		if metadata.status == 'inactive':
			return

		site_path = skill.metadata_path.parent.parent
		changed = False

		if succeeded:
			if metadata.trial_failures:
				metadata.trial_failures = 0  # a success clears the failure streak
				changed = True
			if metadata.status == 'trial':
				metadata.trial_uses += 1
				changed = True
				if metadata.trial_uses >= TRIAL_USES:
					metadata.status = 'stable'
					self._update_manifest_status(site_path, metadata.name, 'stable')
					print(f'[pet-skills] ✅  Skill {metadata.name!r} passed trial — promoted to stable.')
		else:
			metadata.trial_failures += 1
			if metadata.status == 'trial':
				metadata.trial_uses += 1
			changed = True
			if hard_failure or metadata.trial_failures >= TRIAL_MAX_FAILURES:
				metadata.status = 'inactive'
				self._update_manifest_status(site_path, metadata.name, 'inactive')
				reason = 'a hard failure' if hard_failure else f'{metadata.trial_failures} consecutive failures'
				print(f'[pet-skills] ⚠️  Disabled skill {metadata.name!r} after {reason}.')

		if changed:
			skill.metadata_path.write_text(metadata.model_dump_json(indent=2), encoding='utf-8')

	def _update_manifest_status(self, site_path: Path, name: str, status: str) -> None:
		"""Update the skill's status in site.json."""
		manifest_path = site_path / 'site.json'
		if not manifest_path.exists():
			return
		try:
			manifest = SiteSkillManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
			for entry in manifest.skills:
				if entry.name == name:
					entry.status = status  # type: ignore[assignment]
					break
			manifest_path.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')
		except Exception as e:
			print(f'Could not update manifest status for {name!r}: {e}')

	def load_module(self, skill: LoadedSiteSkill) -> ModuleType:
		"""Import a skill module from its file path."""
		assert skill.code_path is not None, 'load_module requires a code-backed skill'
		module_name = f'_pet_site_skill_{skill.metadata.name}_{abs(hash(skill.code_path))}'
		spec = importlib.util.spec_from_file_location(module_name, skill.code_path)
		if spec is None or spec.loader is None:
			raise RuntimeError(f'Could not load skill module at {skill.code_path}')
		module = importlib.util.module_from_spec(spec)
		# Generated code may annotate `ctx: SkillContext` without importing it —
		# provide the names so module exec never fails on annotations.
		module.__dict__['SkillContext'] = SkillContext
		module.__dict__['ActionResult'] = ActionResult
		spec.loader.exec_module(module)
		return module

	def safe_skill_name(self, name: str) -> str:
		"""Return the filesystem-safe skill name used under skills/."""
		return re.sub(r'[^a-zA-Z0-9_-]+', '_', name.strip().lower()).strip('_')

	def capability_text(self, capabilities: dict[str, Any]) -> str:
		"""Describe skill capability limits compactly."""
		if not capabilities:
			return ''
		parts: list[str] = []
		for key, value in sorted(capabilities.items()):
			if isinstance(value, list):
				parts.append(f'{key}={", ".join(str(item) for item in value)}')
			else:
				parts.append(f'{key}={value}')
		return '; '.join(parts)
