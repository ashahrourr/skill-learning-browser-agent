import json
from pathlib import Path

import pytest

from browser_use.pet import PetBridge, PetSession, PetTask
from browser_use.pet_skills import PetSkillRegistry


def _bridge(tmp_path: Path) -> PetBridge:
	return PetBridge(cdp_url=None, max_steps=30, log_dir=tmp_path / 'logs', site_dir=tmp_path / 'sites')


def _write_echo_skill(site_path: Path, *, active: bool = True) -> None:
	(site_path / 'skills').mkdir(parents=True, exist_ok=True)
	(site_path / 'site.json').write_text(
		json.dumps(
			{
				'site': 'example.com',
				'version': 1,
				'skills': [{'name': 'echo_value', 'status': 'stable' if active else 'inactive', 'version': 1}],
			},
			indent=2,
		),
		encoding='utf-8',
	)
	(site_path / 'skills' / 'echo_value.json').write_text(
		json.dumps(
			{
				'name': 'echo_value',
				'status': 'stable',
				'version': 1,
				'description': 'Echo a provided value for test verification.',
				'inputs': ['value'],
				'trigger': 'Use when the test asks to echo a value.',
				'capabilities': {'mode': 'test'},
			},
			indent=2,
		),
		encoding='utf-8',
	)
	(site_path / 'skills' / 'echo_value.py').write_text(
		# Intentionally annotates ctx: SkillContext without importing it — generated
		# skills do this, and the loader must inject the name.
		'async def run(inputs: dict, ctx: SkillContext) -> ActionResult:\n'
		'    return ActionResult(extracted_content=f"echo:{inputs[\'value\']}", include_in_memory=True)\n',
		encoding='utf-8',
	)


def test_site_brain_creation_includes_manifest_skills_memory_and_traces(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='remember this site', url='https://example.com/results')

	memory_path = bridge._ensure_site_memory(task)
	site_path = tmp_path / 'sites' / 'example.com'

	assert memory_path == site_path / 'memory.md'
	assert memory_path.exists()
	assert (site_path / 'site.json').exists()
	assert (site_path / 'skills').is_dir()
	assert (site_path / 'traces').is_dir()
	assert json.loads((site_path / 'site.json').read_text()) == {'site': 'example.com', 'version': 1, 'skills': []}


def test_site_memory_prompt_lists_active_skills(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='echo a value', url='https://example.com/results')
	bridge._ensure_site_memory(task)
	_write_echo_skill(tmp_path / 'sites' / 'example.com')

	prompt = bridge._site_memory_prompt(task)

	assert 'Active site skills available through run_site_skill' in prompt
	assert 'echo_value(value): Echo a provided value for test verification.' in prompt


@pytest.mark.asyncio
async def test_run_active_skill_executes_code_backed_skill(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='echo a value', url='https://example.com/results')
	bridge._ensure_site_memory(task)
	_write_echo_skill(tmp_path / 'sites' / 'example.com')

	result = await bridge._run_active_skill(task, 'echo_value', {'value': 'hello'}, browser_session=None)  # type: ignore[arg-type]

	assert result.error is None
	assert result.extracted_content == 'echo:hello'


@pytest.mark.asyncio
async def test_run_active_skill_rejects_unknown_or_missing_inputs(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='echo a value', url='https://example.com/results')
	bridge._ensure_site_memory(task)
	_write_echo_skill(tmp_path / 'sites' / 'example.com')

	unknown = await bridge._run_active_skill(task, 'missing_skill', {}, browser_session=None)  # type: ignore[arg-type]
	missing_input = await bridge._run_active_skill(task, 'echo_value', {}, browser_session=None)  # type: ignore[arg-type]

	assert unknown.error == "No active site skill named 'missing_skill'."
	assert missing_input.error == "Site skill 'echo_value' missing input(s): value."


def test_inactive_manifest_skill_is_not_loaded(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='echo a value', url='https://example.com/results')
	bridge._ensure_site_memory(task)
	_write_echo_skill(tmp_path / 'sites' / 'example.com', active=False)

	skills = PetSkillRegistry().load_active_skills(tmp_path / 'sites' / 'example.com')

	assert skills == []


@pytest.mark.asyncio
async def test_steps_based_skill_executes_and_updates_trial_counters(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='wait a moment', url='https://example.com/results')
	bridge._ensure_site_memory(task)
	site_path = tmp_path / 'sites' / 'example.com'
	(site_path / 'skills').mkdir(parents=True, exist_ok=True)
	(site_path / 'site.json').write_text(
		json.dumps({'site': 'example.com', 'version': 1, 'skills': [{'name': 'pause', 'status': 'trial', 'version': 1}]}),
		encoding='utf-8',
	)
	(site_path / 'skills' / 'pause.json').write_text(
		json.dumps({'name': 'pause', 'status': 'trial', 'version': 1, 'description': 'Wait briefly.', 'inputs': []}),
		encoding='utf-8',
	)
	(site_path / 'skills' / 'pause.steps.json').write_text(
		json.dumps({'steps': [{'do': 'wait', 'seconds': 0.01}]}),
		encoding='utf-8',
	)

	result = await bridge._run_active_skill(task, 'pause', {}, browser_session=None)  # type: ignore[arg-type]

	assert result.error is None
	assert result.extracted_content == 'done: skill completed all steps'
	metadata = json.loads((site_path / 'skills' / 'pause.json').read_text())
	assert metadata['trial_uses'] == 1
	assert metadata['trial_failures'] == 0


async def test_stable_skill_is_disabled_after_a_hard_failure(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='boom', url='https://example.com/results')
	bridge._ensure_site_memory(task)
	site_path = tmp_path / 'sites' / 'example.com'
	(site_path / 'skills').mkdir(parents=True, exist_ok=True)
	(site_path / 'site.json').write_text(
		json.dumps({'site': 'example.com', 'version': 1, 'skills': [{'name': 'boom', 'status': 'stable', 'version': 1}]}),
		encoding='utf-8',
	)
	# A previously-promoted (stable) skill that now crashes — e.g. the site changed under it.
	(site_path / 'skills' / 'boom.json').write_text(
		json.dumps({'name': 'boom', 'status': 'stable', 'version': 1, 'description': 'x', 'inputs': [], 'entrypoint': 'run'}),
		encoding='utf-8',
	)
	(site_path / 'skills' / 'boom.py').write_text('def run(inputs, ctx):\n\traise RuntimeError("boom")\n', encoding='utf-8')

	result = await bridge._run_active_skill(task, 'boom', {}, browser_session=None)  # type: ignore[arg-type]

	assert result.error is not None
	# Stable skills used to be ignored by the lifecycle; a hard failure must now disable it.
	metadata = json.loads((site_path / 'skills' / 'boom.json').read_text())
	assert metadata['status'] == 'inactive'


def test_successful_trace_still_saves_to_site_traces(tmp_path: Path) -> None:
	bridge = _bridge(tmp_path)
	task = PetTask(session_id='tab:1', task='save trace', url='https://example.com/results')
	session = PetSession(session_id='tab:1')

	bridge._save_site_trace(task, session, 'done', successful=True, history=None)

	trace_files = list((tmp_path / 'sites' / 'example.com' / 'traces').glob('*.json'))
	assert len(trace_files) == 1
	trace = json.loads(trace_files[0].read_text())
	assert trace['version'] == 2
	assert trace['task'] == 'save trace'
	assert trace['successful'] is True
	assert trace['final_result_summary'] == 'done'
	assert 'operations' not in trace
