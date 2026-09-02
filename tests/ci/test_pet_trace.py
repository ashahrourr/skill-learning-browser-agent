import json
from pathlib import Path

from browser_use.pet import PetBridge, PetSession, PetTask
from browser_use.pet_trace import PetTraceBuilder


class _FakeHistory:
	def __init__(self, actions: list[dict]):
		self._actions = actions

	def model_actions(self) -> list[dict]:
		return self._actions


def _click(label: str) -> dict:
	return {'click': {'index': 1}, 'interacted_element': {'ax_name': label, 'attributes': {}}}


def _input(label: str, text: str) -> dict:
	return {'input': {'index': 1, 'text': text}, 'interacted_element': {'ax_name': label, 'attributes': {}}}


def test_trace_operations_keep_failed_attempt_and_successful_retry() -> None:
	builder = PetTraceBuilder()
	history = _FakeHistory(
		[
			_input('drawsStartDate', '2/1/2026'),
			_input('drawsEndDate', '2/28/2026'),
			_click('Update'),
			_click('drawsStartDate'),
			_click('Previous month'),
			_click('Previous month'),
			_click('Previous month'),
			_click('Previous month'),
			_click('02/01/2026'),
			_click('drawsEndDate'),
			_click('02/28/2026'),
			_click('Update'),
			{'extract': {}, 'interacted_element': None},
		]
	)
	visible_steps = [
		'Step 1\nNow: Input the date range for February 2026, then click Update.\n'
		'Action: type "2/1/2026", type "2/28/2026", click #3',
		'Step 2\nLast: The date filter failed because the inputs did not stick.\n'
		'Now: Open the calendar and select February 2026 manually.\n'
		'Action: click #4, click #5, click #5, +6 more',
		'Step 3\nLast: Successfully filtered for February 2026 draws.\nNow: Extract results.\nAction: extract page data',
	]

	operations = builder.operations('find February 2026 draws', history, visible_steps)

	assert [operation.outcome for operation in operations[:2]] == ['failed', 'success']
	assert operations[0].kind == 'ui_action'
	assert operations[0].actions == [
		"type '2/1/2026' into drawsStartDate",
		"type '2/28/2026' into drawsEndDate",
		'click Update',
	]
	assert operations[1].actions == [
		'click drawsStartDate',
		'click Previous month x4',
		'click 02/01/2026',
		'click drawsEndDate',
		'click 02/28/2026',
		'click Update',
	]


def test_saved_trace_v2_has_operations_without_workflow_or_state(tmp_path: Path) -> None:
	bridge = PetBridge(cdp_url=None, max_steps=30, log_dir=tmp_path / 'logs', site_dir=tmp_path / 'sites')
	task = PetTask(session_id='tab:1', task='filter February draws', url='https://example.com/results')
	session = PetSession(session_id='tab:1')
	session.visible_steps = [
		'Step 1\nNow: Click the Update button.\nAction: click #1',
		'Step 2\nLast: Successfully applied the filter.\nNow: Finish.\nAction: finish',
	]
	history = _FakeHistory([_click('Update'), {'done': {'text': 'Done'}, 'interacted_element': None}])

	bridge._save_site_trace(task, session, 'done', successful=True, history=history)

	trace_files = list((tmp_path / 'sites' / 'example.com' / 'traces').glob('*.json'))
	assert len(trace_files) == 1
	trace = json.loads(trace_files[0].read_text())
	assert trace['version'] == 2
	assert 'operations' in trace
	assert trace['operations'][0]['kind'] == 'ui_action'
	assert 'label' not in trace['operations'][0]
	assert 'workflow' not in trace
	serialized = json.dumps(trace)
	assert 'state_before' not in serialized
	assert 'state_after' not in serialized
	assert 'visible_steps' not in serialized


def test_unlabeled_click_label_cannot_be_mistaken_for_a_dom_id() -> None:
	# A bare '#178' fallback once leaked into a learned skill as a fake DOM id.
	builder = PetTraceBuilder()

	label = builder.action_label({'action': 'click', 'target_text': None, 'params': {'index': 178}})

	assert label == 'click [index 178]'


def test_unlabeled_click_label_includes_tag_class_descriptor_when_available() -> None:
	# Anonymous elements (no ax name / aria-label / id) need a real targeting hint,
	# otherwise the skill learner invents accessible names that do not exist.
	builder = PetTraceBuilder()

	label = builder.action_label(
		{'action': 'click', 'target_text': None, 'target_css': 'span.icon-calendar', 'params': {'index': 96}}
	)

	assert label == 'click [index 96] (span.icon-calendar)'


def test_element_fingerprint_records_interaction_time_facts() -> None:
	builder = PetTraceBuilder()

	fingerprint = builder.element_fingerprint(
		{
			'node_name': 'SPAN',
			'attributes': {'class': 'icon-calendar extra', 'title': 'pick a date'},
			'x_path': 'html/body/div/span[2]',
		}
	)

	assert fingerprint == {
		'tag': 'span',
		'classes': ['icon-calendar', 'extra'],
		'label': 'pick a date',
		'x_path': 'html/body/div/span[2]',
	}
	assert builder.element_fingerprint(None) is None


def test_operations_carry_element_fingerprints_for_ui_actions() -> None:
	builder = PetTraceBuilder()
	history = _FakeHistory(
		[
			{
				'click': {'index': 96},
				'interacted_element': {
					'node_name': 'SPAN',
					'attributes': {'class': 'icon-calendar'},
					'x_path': 'html/body/div/span[1]',
				},
			},
			{
				'click': {'index': 98},
				'interacted_element': {
					'node_name': 'SPAN',
					'attributes': {'class': 'icon-calendar'},
					'x_path': 'html/body/div/span[2]',
				},
			},
		]
	)
	visible_steps = ['Step 1\nNow: Open both date pickers.\nAction: click #1, click #2']

	operations = builder.operations('filter dates', history, visible_steps)

	assert len(operations) == 1
	# Twins share tag+classes but keep distinct x_paths so the learner can assign nth.
	assert operations[0].elements == [
		{'tag': 'span', 'classes': ['icon-calendar'], 'x_path': 'html/body/div/span[1]'},
		{'tag': 'span', 'classes': ['icon-calendar'], 'x_path': 'html/body/div/span[2]'},
	]


def test_interacted_element_descriptor_uses_tag_and_first_class() -> None:
	builder = PetTraceBuilder()

	assert (
		builder.interacted_element_descriptor({'node_name': 'SPAN', 'attributes': {'class': 'icon-calendar extra'}})
		== 'span.icon-calendar'
	)
	assert builder.interacted_element_descriptor({'node_name': 'BUTTON', 'attributes': {}}) == 'button'
	assert builder.interacted_element_descriptor(None) is None


def test_trace_text_fields_are_capped() -> None:
	builder = PetTraceBuilder()
	long_text = 'x' * 1000
	history = _FakeHistory([_click('Submit')])
	visible_steps = [f'Step 1\nNow: {long_text}\nAction: click #1']

	operation = builder.operations('task', history, visible_steps)[0]

	assert len(builder.final_result_summary(f'{long_text} Attachments: secret')) == 240
	assert len(operation.goal) == 300
	assert operation.result == ''


def test_snake_move_clicks_are_operate_site_not_verify_result() -> None:
	builder = PetTraceBuilder()
	history = _FakeHistory(
		[
			_click('Move right'),
			_click('Move right'),
			{'done': {'text': 'Target reached'}, 'interacted_element': None},
		]
	)
	visible_steps = [
		'Step 1\nNow: Click Right button to move snake head from column 5 to column 6 and eat the food.\nAction: click #1',
		'Step 2\nLast: Successfully clicked Right button - snake head moved from (5,5) to (5,6), food still at (5,8). Need two more right moves.\n'
		'Now: Click Right button to move snake head from column 6 to column 7, getting closer to food at (5,8).\n'
		'Action: click #1',
		'Step 3\nLast: Successfully clicked Right - snake head moved from (5,6) to (5,7), food still at (5,8). Need one more right move to eat.\n'
		'Now: Target has been reached - call done to report completion.\n'
		'Action: finish',
	]

	operations = builder.operations('reach target', history, visible_steps)

	assert [operation.kind for operation in operations] == ['ui_action', 'ui_action', 'finish_action']


def test_operation_kind_is_action_based_not_task_text_based() -> None:
	builder = PetTraceBuilder()
	history = _FakeHistory(
		[
			{'scroll': {'down': True, 'pages': 1}, 'interacted_element': None},
			{'read_file': {'file_name': 'results.md'}, 'interacted_element': None},
		]
	)
	visible_steps = [
		'Step 1\nNow: Scroll down to reveal all February 2026 draws, then extract all draw data.\nAction: scroll down',
		'Step 2\nLast: Successfully filtered for February 2026 draws.\n'
		'Now: Read current results.md, update with complete final analysis, then call done to report results.\n'
		'Action: read_file',
	]

	operations = builder.operations(
		'find the most common number for each position for the month of feb 2026',
		history,
		visible_steps,
	)

	assert [operation.kind for operation in operations] == ['ui_action', 'file_action']
