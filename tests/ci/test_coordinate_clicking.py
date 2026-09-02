"""Tests for coordinate clicking feature.

This feature allows certain models (Claude Sonnet 4, Claude Opus 4, Gemini 3 Pro, browser-use/* models)
to use coordinate-based clicking, while other models only get index-based clicking.
"""

import pytest

from browser_use.tools.service import Tools
from browser_use.tools.views import ClickElementAction, ClickElementActionIndexOnly


class TestCoordinateClickingTools:
	"""Test the Tools class coordinate clicking functionality."""

	def test_default_coordinate_clicking_disabled(self):
		"""By default, coordinate clicking should be disabled."""
		tools = Tools()

		assert tools._coordinate_clicking_enabled is False

	def test_default_uses_index_only_action(self):
		"""Default Tools should use ClickElementActionIndexOnly."""
		tools = Tools()

		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		assert click_action.param_model == ClickElementActionIndexOnly

	def test_default_click_schema_has_only_index(self):
		"""Default click action schema should only have index property."""
		tools = Tools()

		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		schema = click_action.param_model.model_json_schema()

		assert 'index' in schema['properties']
		assert 'coordinate_x' not in schema['properties']
		assert 'coordinate_y' not in schema['properties']

	def test_enable_coordinate_clicking(self):
		"""Enabling coordinate clicking should switch to ClickElementAction."""
		tools = Tools()
		tools.set_coordinate_clicking(True)

		assert tools._coordinate_clicking_enabled is True

		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		assert click_action.param_model == ClickElementAction

	def test_enabled_click_schema_has_coordinates(self):
		"""Enabled click action schema should have index and coordinate properties."""
		tools = Tools()
		tools.set_coordinate_clicking(True)

		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		schema = click_action.param_model.model_json_schema()

		assert 'index' in schema['properties']
		assert 'coordinate_x' in schema['properties']
		assert 'coordinate_y' in schema['properties']

	def test_disable_coordinate_clicking(self):
		"""Disabling coordinate clicking should switch back to index-only."""
		tools = Tools()
		tools.set_coordinate_clicking(True)
		tools.set_coordinate_clicking(False)

		assert tools._coordinate_clicking_enabled is False

		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		assert click_action.param_model == ClickElementActionIndexOnly

	def test_set_coordinate_clicking_idempotent(self):
		"""Setting the same value twice should not cause issues."""
		tools = Tools()

		# Enable twice
		tools.set_coordinate_clicking(True)
		tools.set_coordinate_clicking(True)
		assert tools._coordinate_clicking_enabled is True

		# Disable twice
		tools.set_coordinate_clicking(False)
		tools.set_coordinate_clicking(False)
		assert tools._coordinate_clicking_enabled is False

	def test_schema_title_consistent(self):
		"""Schema title should be 'ClickElementAction' regardless of mode."""
		tools = Tools()

		# Check default (disabled)
		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		schema = click_action.param_model.model_json_schema()
		assert schema['title'] == 'ClickElementAction'

		# Check enabled
		tools.set_coordinate_clicking(True)
		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		schema = click_action.param_model.model_json_schema()
		assert schema['title'] == 'ClickElementAction'


class TestDragAction:
	"""The drag action is registered in lockstep with coordinate clicking."""

	def test_drag_absent_by_default(self):
		tools = Tools()
		assert 'drag' not in tools.registry.registry.actions

	def test_drag_registered_when_coordinates_enabled(self):
		tools = Tools()
		tools.set_coordinate_clicking(True)
		drag_action = tools.registry.registry.actions.get('drag')
		assert drag_action is not None
		schema = drag_action.param_model.model_json_schema()
		assert 'path' in schema['properties']

	def test_drag_removed_when_coordinates_disabled(self):
		tools = Tools()
		tools.set_coordinate_clicking(True)
		tools.set_coordinate_clicking(False)
		assert 'drag' not in tools.registry.registry.actions

	def test_drag_path_requires_two_points(self):
		from pydantic import ValidationError

		from browser_use.tools.views import DragAction

		with pytest.raises(ValidationError):
			DragAction(path=[(1, 1)])
		# two points is the minimum valid drag
		assert len(DragAction(path=[(1, 1), (2, 2)]).path) == 2


class TestCoordinateClickingModelDetection:
	"""Test the model detection logic for coordinate clicking."""

	@pytest.mark.parametrize(
		'model_name,expected_coords',
		[
			# Models that SHOULD have coordinate clicking (claude-sonnet-4*, claude-opus-4*, gemini-3-pro*, browser-use/*)
			('claude-sonnet-4-5', True),
			('claude-sonnet-4-5-20250101', True),
			('claude-sonnet-4-0', True),
			('claude-sonnet-4', True),
			('claude-opus-4-5', True),
			('claude-opus-4-5-latest', True),
			('claude-opus-4-0', True),
			('claude-opus-4', True),
			('gemini-3-pro-preview', True),
			('gemini-3-pro', True),
			('browser-use/fast', True),
			('browser-use/accurate', True),
			('CLAUDE-SONNET-4-5', True),  # Case insensitive
			('CLAUDE-SONNET-4', True),  # Case insensitive
			('GEMINI-3-PRO', True),  # Case insensitive
			# Models that should NOT have coordinate clicking
			('claude-3-5-sonnet', False),
			('claude-sonnet-3-5', False),
			('gpt-4o', False),
			('gpt-4-turbo', False),
			('gemini-2.0-flash', False),
			('gemini-1.5-pro', False),
			('llama-3.1-70b', False),
			('mistral-large', False),
		],
	)
	def test_model_detection_patterns(self, model_name: str, expected_coords: bool):
		"""Test that the model detection patterns correctly identify coordinate-capable models."""
		model_lower = model_name.lower()
		supports_coords = any(
			pattern in model_lower for pattern in ['claude-sonnet-4', 'claude-opus-4', 'gemini-3-pro', 'browser-use/']
		)
		assert supports_coords == expected_coords, f'Model {model_name}: expected {expected_coords}, got {supports_coords}'


class TestCoordinateClickingWithPassedTools:
	"""Test that coordinate clicking works correctly when Tools is passed to Agent."""

	def test_tools_can_be_modified_after_creation(self):
		"""Tools created externally can have coordinate clicking enabled."""
		tools = Tools()
		assert tools._coordinate_clicking_enabled is False

		# Simulate what Agent does for coordinate-capable models
		tools.set_coordinate_clicking(True)

		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		assert click_action.param_model == ClickElementAction

	def test_tools_state_preserved_after_modification(self):
		"""Verify that other tool state is preserved when toggling coordinate clicking."""
		tools = Tools(exclude_actions=['search'])

		# Search should be excluded
		assert 'search' not in tools.registry.registry.actions

		# Enable coordinate clicking
		tools.set_coordinate_clicking(True)

		# Search should still be excluded
		assert 'search' not in tools.registry.registry.actions

		# Click should have coordinates
		click_action = tools.registry.registry.actions.get('click')
		assert click_action is not None
		assert click_action.param_model == ClickElementAction
