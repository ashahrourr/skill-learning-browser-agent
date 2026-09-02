import pytest

from browser_use.pet_skill_context import SkillContext
from browser_use.pet_skill_steps import (
	SkillStep,
	SkillSteps,
	SkillStepTarget,
	render_template,
	run_skill_steps,
	validate_generated_steps,
)


def test_render_template_substitutes_plain_inputs() -> None:
	assert render_template('search for {q} now', {'q': 'mega millions'}) == 'search for mega millions now'


def test_render_template_reformats_dates_from_any_common_format() -> None:
	inputs_iso = {'start_date': '2024-04-01'}
	inputs_us = {'start_date': '04/01/2024'}
	inputs_words = {'start_date': 'April 1, 2024'}
	for inputs in (inputs_iso, inputs_us, inputs_words):
		assert render_template('{start_date|%m/%d/%Y}', inputs) == '04/01/2024'
		assert render_template('{start_date|%Y}', inputs) == '2024'


def test_render_template_errors_on_missing_input_and_bad_date() -> None:
	with pytest.raises(KeyError):
		render_template('{missing}', {})
	with pytest.raises(ValueError):
		render_template('{start_date|%Y}', {'start_date': 'not a date'})


async def test_run_skill_steps_executes_wait_steps_without_browser() -> None:
	skill = SkillSteps(steps=[SkillStep(do='wait', seconds=0.01), SkillStep(do='wait', seconds=0.01)])

	result = await run_skill_steps(skill, {}, ctx=SkillContext(browser_session=None))  # type: ignore[arg-type]

	assert result.error is None
	assert result.extracted_content == 'done: skill completed all steps'


async def test_run_skill_steps_returns_step_error_instead_of_raising() -> None:
	skill = SkillSteps(steps=[SkillStep(do='navigate')])  # url missing → step error

	result = await run_skill_steps(skill, {}, ctx=SkillContext(browser_session=None))  # type: ignore[arg-type]

	assert result.error is not None
	assert 'step 1' in result.error.lower()
	assert 'navigate' in result.error


def test_validate_generated_steps_flags_numeric_ids_and_unknown_placeholders() -> None:
	# Reproduces the megamillions failure: the LLM stored browser-use element
	# indices (178/180) as DOM ids and used a placeholder the skill never declared.
	skill = SkillSteps(
		steps=[
			SkillStep(do='click', target=SkillStepTarget(id='178')),
			SkillStep(do='select', target=SkillStepTarget(aria='Select a year'), value='{start_date|%Y}'),
			SkillStep(
				do='probe_until',
				until_aria_exists='{month_name} 01, {year}',
				repeat_click=SkillStepTarget(id='180'),
			),
			SkillStep(do='click', target=SkillStepTarget(css='#181')),
		]
	)

	problems = validate_generated_steps(skill, ['month_name', 'year', 'last_day'])

	assert any("'178'" in problem and 'step 1' in problem for problem in problems)
	assert any("'180'" in problem and 'step 3' in problem for problem in problems)
	assert any("'#181'" in problem and 'step 4' in problem for problem in problems)
	assert any('start_date' in problem for problem in problems)
	# Declared inputs used as placeholders are fine.
	assert not any('{month_name}' in problem or '{year}' in problem for problem in problems)


def test_validate_generated_steps_accepts_clean_skill() -> None:
	skill = SkillSteps(
		steps=[
			SkillStep(do='navigate', url='https://example.com/results'),
			SkillStep(do='click', target=SkillStepTarget(id='drawsStartDate')),
			SkillStep(do='select', target=SkillStepTarget(aria='Select a year'), value='{year}'),
			SkillStep(
				do='probe_until',
				until_aria_exists='{month_name} 01, {year}',
				repeat_click=SkillStepTarget(aria='Previous month'),
			),
			SkillStep(do='click', target=SkillStepTarget(text='Update')),
		]
	)

	assert validate_generated_steps(skill, ['month_name', 'year']) == []


def test_validate_generated_steps_rejects_targets_absent_from_trace_elements() -> None:
	# Reproduces the second megamillions failure: the LLM invented aria 'Start Date'
	# from goal prose; no recorded element ever had that name.
	trace_elements = [
		{'tag': 'span', 'classes': ['icon-calendar'], 'x_path': 'html/body/div/span[1]'},
		{'tag': 'select', 'aria': 'Select a year'},
		{'tag': 'a', 'aria': 'Previous month'},
		{'tag': 'input', 'id': 'drawsStartDate'},
		{'tag': 'button', 'text': 'Update'},
	]
	invented = SkillSteps(steps=[SkillStep(do='click', target=SkillStepTarget(aria='Start Date'))])
	problems = validate_generated_steps(invented, [], trace_elements=trace_elements)
	assert len(problems) == 1
	assert "'Start Date'" in problems[0] and 'recorded' in problems[0]

	grounded = SkillSteps(
		steps=[
			SkillStep(do='click', target=SkillStepTarget(css='span.icon-calendar', nth=1)),
			SkillStep(do='select', target=SkillStepTarget(aria='Select a year'), value='{year}'),
			SkillStep(
				do='probe_until',
				until_aria_exists='{month_name} 01, {year}',
				repeat_click=SkillStepTarget(aria='Previous month'),
			),
			SkillStep(do='click', target=SkillStepTarget(id='drawsStartDate')),
			SkillStep(do='click', target=SkillStepTarget(text='Update')),
		]
	)
	assert validate_generated_steps(grounded, ['month_name', 'year'], trace_elements=trace_elements) == []


def test_validate_generated_steps_accepts_compound_class_selectors_that_cooccur() -> None:
	# Real failure: LLM chose 'span.fi.icon_calendar' — both classes recorded on one
	# element — but the validator's css grammar only knew single-class selectors.
	trace_elements = [
		{'tag': 'span', 'classes': ['fi', 'icon_calendar', 'icon_pastStart']},
		{'tag': 'button', 'classes': ['btn-update']},
	]

	grounded = SkillSteps(steps=[SkillStep(do='click', target=SkillStepTarget(css='span.fi.icon_calendar'))])
	assert validate_generated_steps(grounded, [], trace_elements=trace_elements) == []

	# Classes that exist only on *different* elements must not pass via union.
	frankenstein = SkillSteps(steps=[SkillStep(do='click', target=SkillStepTarget(css='span.fi.btn-update'))])
	problems = validate_generated_steps(frankenstein, [], trace_elements=trace_elements)
	assert len(problems) == 1 and 'span.fi.btn-update' in problems[0]


def test_validate_generated_steps_skips_provenance_when_no_elements_recorded() -> None:
	# Old traces predate fingerprints; provenance cannot be enforced against them.
	skill = SkillSteps(steps=[SkillStep(do='click', target=SkillStepTarget(aria='Start Date'))])

	assert validate_generated_steps(skill, [], trace_elements=[]) == []
	assert validate_generated_steps(skill, []) == []


def test_target_nth_disambiguates_identical_elements_and_rejects_invalid_values() -> None:
	target = SkillStepTarget.model_validate({'css': 'span.icon-calendar', 'nth': 2})
	assert target.nth == 2
	skill = SkillSteps(steps=[SkillStep(do='click', target=target)])
	assert validate_generated_steps(skill, []) == []
	with pytest.raises(Exception):
		SkillStepTarget.model_validate({'css': 'span.icon-calendar', 'nth': 0})


def test_schema_rejects_unknown_fields_and_unknown_step_kinds() -> None:
	with pytest.raises(Exception):
		SkillStep.model_validate({'do': 'click', 'selector_chain': '#a + span'})
	with pytest.raises(Exception):
		SkillStep.model_validate({'do': 'execute_javascript'})
	with pytest.raises(Exception):
		SkillStepTarget.model_validate({'xpath': '//div'})
