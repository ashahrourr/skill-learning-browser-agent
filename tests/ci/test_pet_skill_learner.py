import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from browser_use.pet_skill_learner import PetSkillLearner, SavedSkillProposal, SiteSkillProposal
from browser_use.pet_skill_steps import SkillStep, SkillSteps, SkillStepTarget


class _FakeLLM:
	def __init__(self, proposal: SiteSkillProposal):
		self.proposal = proposal
		self.messages = None

	async def ainvoke(self, messages, output_format):
		self.messages = messages
		assert output_format is SiteSkillProposal
		return SimpleNamespace(completion=self.proposal)


def _write_trace(
	site_path: Path, name: str, *, successful: bool = True, version: int = 2, elements: list[dict] | None = None
) -> None:
	trace_dir = site_path / 'traces'
	trace_dir.mkdir(parents=True, exist_ok=True)
	operation = {
		'kind': 'ui_action',
		'goal': 'Set date range.',
		'result': 'Date filter worked.',
		'outcome': 'success',
		'actions': ['click Start Date', 'click 02/01/2026', 'click Update'],
	}
	if elements is not None:
		operation['elements'] = elements
	(trace_dir / name).write_text(
		json.dumps(
			{
				'version': version,
				'successful': successful,
				'url': 'https://example.com/results',
				'task': 'filter February results',
				'final_result_summary': 'Filtered results.',
				'operations': [operation],
			},
			indent=2,
		),
		encoding='utf-8',
	)


def test_load_recent_traces_uses_successful_v2_operation_traces_only(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(site_path, '20260609-good.json')
	_write_trace(site_path, '20260609-failed.json', successful=False)
	_write_trace(site_path, '20260609-old.json', version=1)

	traces = PetSkillLearner(tmp_path / 'sites').load_recent_traces('example.com')

	assert len(traces) == 1
	assert traces[0]['task'] == 'filter February results'
	assert traces[0]['operations'][0]['kind'] == 'ui_action'


@pytest.mark.asyncio
async def test_propose_saves_proposal_without_creating_skill_files(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(site_path, '20260609-good.json')
	llm = _FakeLLM(
		SiteSkillProposal(
			worth_building=True,
			name='set_date_range',
			description='Set the visible date range filter.',
			inputs=['start_date', 'end_date'],
			trigger='Use on the Previous Drawings page when filtering by date range.',
			evidence=['Direct typing failed, calendar selection worked.'],
			suggested_steps=['Open start date calendar.', 'Select start and end dates.', 'Click Update.'],
			safety_notes=['Operate the live UI only.'],
		)
	)

	saved = await PetSkillLearner(tmp_path / 'sites').propose('example.com', llm)

	assert saved.proposal.name == 'set_date_range'
	proposals = list((site_path / 'skill_proposals').glob('*.json'))
	assert len(proposals) == 1
	assert json.loads(proposals[0].read_text())['proposal']['name'] == 'set_date_range'
	assert not (site_path / 'skills' / 'set_date_range.py').exists()
	assert not (site_path / 'skills' / 'set_date_range.json').exists()


async def test_propose_passes_existing_skills_into_prompt_for_dedup(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(site_path, '20260609-good.json')
	llm = _FakeLLM(SiteSkillProposal(worth_building=False))

	await PetSkillLearner(tmp_path / 'sites').propose(
		'example.com',
		llm,
		existing_skills=['set_date_range(start_date, end_date): Set the visible date range filter.'],
	)

	prompt_text = ' '.join(getattr(m, 'content', '') or '' for m in llm.messages)
	# The existing skill must be visible to the proposer so it can avoid duplicating it.
	assert 'set_date_range(start_date, end_date)' in prompt_text
	assert 'Reconcile against them' in prompt_text or 'reconcile' in prompt_text.lower()


@pytest.mark.asyncio
async def test_generate_skill_code_writes_valid_python_and_registers_trial_skill(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(site_path, '20260609-good.json')
	learner = PetSkillLearner(tmp_path / 'sites')
	proposal = SavedSkillProposal(
		created_at='2026-06-11T00:00:00+00:00',
		site='example.com',
		trace_count=1,
		proposal=SiteSkillProposal(
			worth_building=True,
			name='set_date_range',
			description='Set the visible date range filter.',
			inputs=['start_date', 'end_date'],
			trigger='Use when filtering by date range.',
		),
	)

	class _FakeCodeLLM:
		"""Mimics the runtime response shape: code in .completion, wrapped in markdown fences."""

		async def ainvoke(self, messages):
			code = (
				'```python\n'
				'from browser_use import ActionResult\n\n\n'
				'async def run(inputs: dict, ctx) -> ActionResult:\n'
				"\treturn ActionResult(extracted_content='done')\n"
				'```'
			)
			return SimpleNamespace(completion=code, thinking=None, usage=None)

	code_path = await learner.generate_skill_code(proposal, _FakeCodeLLM())

	source = code_path.read_text(encoding='utf-8')
	assert 'completion=' not in source
	assert '```' not in source
	compile(source, str(code_path), 'exec')
	metadata = json.loads((site_path / 'skills' / 'set_date_range.json').read_text())
	assert metadata['status'] == 'trial'
	manifest = json.loads((site_path / 'site.json').read_text())
	assert manifest['skills'] == [{'name': 'set_date_range', 'status': 'trial', 'version': 1}]


@pytest.mark.asyncio
async def test_generate_skill_steps_writes_steps_file_and_registers_trial_skill(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(site_path, '20260609-good.json')
	learner = PetSkillLearner(tmp_path / 'sites')
	proposal = SavedSkillProposal(
		created_at='2026-06-11T00:00:00+00:00',
		site='example.com',
		trace_count=1,
		proposal=SiteSkillProposal(
			worth_building=True,
			name='set_date_range',
			description='Set the visible date range filter.',
			inputs=['start_date', 'end_date'],
			trigger='Use when filtering by date range.',
		),
	)
	generated = SkillSteps(
		steps=[
			SkillStep(do='click', target=SkillStepTarget(id='drawsStartDate')),
			SkillStep(do='select', target=SkillStepTarget(aria='Select a year'), value='{start_date|%Y}'),
			SkillStep(
				do='probe_until',
				until_aria_exists='{start_date|%m/%d/%Y}',
				repeat_click=SkillStepTarget(aria='Previous month'),
			),
			SkillStep(do='click', target=SkillStepTarget(aria='{start_date|%m/%d/%Y}')),
			SkillStep(do='click', target=SkillStepTarget(text='Update')),
		]
	)

	class _FakeStepsLLM:
		async def ainvoke(self, messages, output_format):
			assert output_format is SkillSteps
			return SimpleNamespace(completion=generated)

	steps_path = await learner.generate_skill_steps(proposal, _FakeStepsLLM())

	assert steps_path == site_path / 'skills' / 'set_date_range.steps.json'
	reloaded = SkillSteps.model_validate_json(steps_path.read_text())
	assert len(reloaded.steps) == 5
	metadata = json.loads((site_path / 'skills' / 'set_date_range.json').read_text())
	assert metadata['status'] == 'trial'
	manifest = json.loads((site_path / 'site.json').read_text())
	assert manifest['skills'] == [{'name': 'set_date_range', 'status': 'trial', 'version': 1}]


def _date_range_proposal() -> SavedSkillProposal:
	return SavedSkillProposal(
		created_at='2026-06-11T00:00:00+00:00',
		site='example.com',
		trace_count=1,
		proposal=SiteSkillProposal(
			worth_building=True,
			name='set_date_range',
			description='Set the visible date range filter.',
			inputs=['month_name', 'year'],
			trigger='Use when filtering by date range.',
		),
	)


_BAD_STEPS = SkillSteps(
	steps=[
		SkillStep(do='click', target=SkillStepTarget(id='178')),
		SkillStep(do='select', target=SkillStepTarget(aria='Select a year'), value='{start_date|%Y}'),
	]
)

_GOOD_STEPS = SkillSteps(
	steps=[
		SkillStep(do='click', target=SkillStepTarget(aria='Start Date calendar icon')),
		SkillStep(do='select', target=SkillStepTarget(aria='Select a year'), value='{year}'),
	]
)


@pytest.mark.asyncio
async def test_generate_skill_steps_retries_with_validation_feedback_then_saves_corrected(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(site_path, '20260609-good.json')
	learner = PetSkillLearner(tmp_path / 'sites')

	class _RetryLLM:
		def __init__(self):
			self.calls: list[list] = []

		async def ainvoke(self, messages, output_format):
			self.calls.append(messages)
			return SimpleNamespace(completion=_BAD_STEPS if len(self.calls) == 1 else _GOOD_STEPS)

	llm = _RetryLLM()
	steps_path = await learner.generate_skill_steps(_date_range_proposal(), llm)

	assert len(llm.calls) == 2
	correction = str(llm.calls[1][-1].content)
	assert "'178'" in correction
	assert 'start_date' in correction
	reloaded = SkillSteps.model_validate_json(steps_path.read_text())
	assert reloaded == _GOOD_STEPS


@pytest.mark.asyncio
async def test_generate_skill_steps_enforces_provenance_against_trace_elements(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(
		site_path,
		'20260609-good.json',
		elements=[
			{'tag': 'span', 'classes': ['icon-calendar'], 'x_path': 'html/body/div/span[1]'},
			{'tag': 'button', 'text': 'Update'},
		],
	)
	learner = PetSkillLearner(tmp_path / 'sites')
	invented = SkillSteps(steps=[SkillStep(do='click', target=SkillStepTarget(aria='Start Date'))])
	grounded = SkillSteps(steps=[SkillStep(do='click', target=SkillStepTarget(css='span.icon-calendar'))])

	class _InventsThenGrounds:
		def __init__(self):
			self.calls: list[list] = []

		async def ainvoke(self, messages, output_format):
			self.calls.append(messages)
			return SimpleNamespace(completion=invented if len(self.calls) == 1 else grounded)

	llm = _InventsThenGrounds()
	steps_path = await learner.generate_skill_steps(_date_range_proposal(), llm)

	assert len(llm.calls) == 2
	assert "'Start Date'" in str(llm.calls[1][-1].content)
	assert SkillSteps.model_validate_json(steps_path.read_text()) == grounded


@pytest.mark.asyncio
async def test_generate_skill_steps_raises_and_writes_nothing_when_steps_stay_invalid(tmp_path: Path) -> None:
	site_path = tmp_path / 'sites' / 'example.com'
	_write_trace(site_path, '20260609-good.json')
	learner = PetSkillLearner(tmp_path / 'sites')

	class _AlwaysBadLLM:
		async def ainvoke(self, messages, output_format):
			return SimpleNamespace(completion=_BAD_STEPS)

	with pytest.raises(ValueError, match='178'):
		await learner.generate_skill_steps(_date_range_proposal(), _AlwaysBadLLM())

	assert not (site_path / 'skills').exists()
	assert not (site_path / 'site.json').exists()


@pytest.mark.asyncio
async def test_propose_without_traces_returns_no_skill_and_does_not_call_llm(tmp_path: Path) -> None:
	llm = _FakeLLM(SiteSkillProposal(worth_building=True, name='should_not_be_used'))

	saved = await PetSkillLearner(tmp_path / 'sites').propose('missing.example', llm)

	assert saved.trace_count == 0
	assert saved.proposal.worth_building is False
	assert llm.messages is None
	assert not (tmp_path / 'sites' / 'missing.example' / 'skill_proposals').exists()
