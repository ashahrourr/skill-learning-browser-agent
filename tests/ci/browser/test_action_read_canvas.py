"""Real-browser test for the read_canvas action.

Verifies that read_canvas returns the exact pixel grid of a <canvas> (in its intrinsic
drawing resolution, grouped by color) so an agent can faithfully reproduce a drawing
instead of guessing from a lossy screenshot.
"""

import json

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile, ViewportSize
from browser_use.tools.service import Tools
from browser_use.tools.views import ReadCanvasAction

# A 4x4 canvas (displayed larger) with three known opaque pixels and the rest transparent.
CANVAS_HTML = """
<html><body style="margin:0">
	<canvas id="art" width="4" height="4" style="width:160px;height:160px;image-rendering:pixelated"></canvas>
	<script>
		const ctx = document.getElementById('art').getContext('2d');
		// red at (0,0), green at (1,2), blue at (3,3); everything else transparent.
		ctx.fillStyle = '#ff0000'; ctx.fillRect(0, 0, 1, 1);
		ctx.fillStyle = '#00ff00'; ctx.fillRect(1, 2, 1, 1);
		ctx.fillStyle = '#0000ff'; ctx.fillRect(3, 3, 1, 1);
	</script>
</body></html>
"""


# Mimics Piskel: a small thumbnail canvas FIRST in the DOM (always has the drawing) and a large
# editable "drawing" canvas after it (here left blank). The old code grabbed the first canvas (the
# thumbnail) and could never see the big canvas; the fix should pick the big one by default.
STACKED_HTML = """
<html><body style="margin:0">
	<canvas class="thumb" width="4" height="4" style="width:32px;height:32px"></canvas>
	<canvas class="drawing-surface" width="8" height="8" style="width:320px;height:320px"></canvas>
	<script>
		// thumbnail has 2 pixels of content; the big drawing surface is blank.
		const t = document.querySelector('.thumb').getContext('2d');
		t.fillStyle = '#ff0000'; t.fillRect(0, 0, 1, 1);
		t.fillStyle = '#00ff00'; t.fillRect(2, 2, 1, 1);
	</script>
</body></html>
"""


@pytest.fixture(scope='module')
def canvas_server():
	server = HTTPServer()
	server.start()
	server.expect_request('/art').respond_with_data(CANVAS_HTML, content_type='text/html')
	server.expect_request('/stacked').respond_with_data(STACKED_HTML, content_type='text/html')
	yield server
	server.stop()


@pytest.fixture
async def browser_session():
	session = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=True,
			window_size=ViewportSize(width=800, height=600),
		)
	)
	await session.start()
	yield session
	await session.kill()


async def test_read_canvas_returns_exact_pixels(browser_session, canvas_server):
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/art')

	tools = Tools()
	result = await tools.act(
		action=tools.registry.create_action_model()(read_canvas=ReadCanvasAction(selector='#art')),
		browser_session=browser_session,
	)
	assert result.error is None, result.error
	data = json.loads(result.extracted_content)

	# Intrinsic drawing grid, not the 160x160 CSS display size.
	assert data['width'] == 4 and data['height'] == 4
	assert data['opaque_pixels'] == 3
	assert data['colors']['#ff0000'] == [[0, 0]]
	assert data['colors']['#00ff00'] == [[1, 2]]
	assert data['colors']['#0000ff'] == [[3, 3]]
	# On-screen box is reported so the agent can map grid pixels to click coordinates.
	assert data['screen']['width'] == 160 and data['screen']['height'] == 160


async def test_read_canvas_respects_max_pixels_cap(browser_session, canvas_server):
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/art')
	tools = Tools()
	result = await tools.act(
		action=tools.registry.create_action_model()(read_canvas=ReadCanvasAction(selector='#art', max_pixels=2)),
		browser_session=browser_session,
	)
	data = json.loads(result.extracted_content)
	assert data['opaque_pixels'] == 2
	assert data['capped'] is True


async def test_read_canvas_no_args_lists_all_without_reading(browser_session, canvas_server):
	"""First call (no selector/index) hands back the full canvas list and reads no pixels."""
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/stacked')
	tools = Tools()
	result = await tools.act(
		action=tools.registry.create_action_model()(read_canvas=ReadCanvasAction()),
		browser_session=browser_session,
	)
	assert result.error is None, result.error
	data = json.loads(result.extracted_content)

	assert data['needs_selection'] is True
	assert 'colors' not in data  # no pixels read yet
	assert len(data['canvases']) == 2
	# Each canvas is described well enough for the model to choose (size + content count).
	thumb = next(c for c in data['canvases'] if c['cls'] == 'thumb')
	surface = next(c for c in data['canvases'] if c['cls'] == 'drawing-surface')
	assert thumb['opaque_pixels'] == 2 and thumb['width'] == 4
	assert surface['opaque_pixels'] == 0 and surface['width'] == 8


async def test_read_canvas_pick_by_index(browser_session, canvas_server):
	"""After listing, the model picks a specific canvas by index and gets its pixels."""
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/stacked')
	tools = Tools()
	listing = json.loads(
		(
			await tools.act(
				action=tools.registry.create_action_model()(read_canvas=ReadCanvasAction()),
				browser_session=browser_session,
			)
		).extracted_content
	)
	thumb_index = next(c['index'] for c in listing['canvases'] if c['cls'] == 'thumb')

	result = await tools.act(
		action=tools.registry.create_action_model()(read_canvas=ReadCanvasAction(index=thumb_index)),
		browser_session=browser_session,
	)
	data = json.loads(result.extracted_content)
	assert data['chosen_index'] == thumb_index
	assert data['width'] == 4 and data['opaque_pixels'] == 2
	assert data['colors']['#ff0000'] == [[0, 0]]


async def test_read_canvas_ambiguous_selector_returns_list(browser_session, canvas_server):
	"""A selector matching multiple canvases returns the list instead of guessing."""
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/stacked')
	tools = Tools()
	result = await tools.act(
		action=tools.registry.create_action_model()(read_canvas=ReadCanvasAction(selector='canvas')),
		browser_session=browser_session,
	)
	data = json.loads(result.extracted_content)
	assert data['needs_selection'] is True
	assert data['matched'] == 2
	assert 'colors' not in data


async def test_read_canvas_missing_canvas_errors(browser_session, canvas_server):
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/art')
	tools = Tools()
	result = await tools.act(
		action=tools.registry.create_action_model()(read_canvas=ReadCanvasAction(selector='#nope')),
		browser_session=browser_session,
	)
	assert result.error is not None
	assert 'no canvas' in result.error.lower()
