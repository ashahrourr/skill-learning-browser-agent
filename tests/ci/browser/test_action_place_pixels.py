"""Real-browser test for the place_pixels action.

Verifies that place_pixels coalesces a color's cells into contiguous horizontal pen strokes,
maps native grid cells to their on-screen tile centers, and paints them onto a canvas — so a
sprite fills in as fast sweeps rather than one slow click per pixel.
"""

import json

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile, ViewportSize
from browser_use.tools.service import Tools
from browser_use.tools.views import PlacePixelsAction

# A canvas that records stroke starts (pointerdown) and every pressed point (down + move-while-held),
# in canvas-local CSS pixels, so the test can check both stroke count and pixel coverage.
CANVAS_HTML = """
<html><body style="margin:0">
	<canvas id="pad" width="4" height="4" style="width:160px;height:160px;position:absolute;left:40px;top:20px"></canvas>
	<script>
		window.__downs = [];
		window.__points = [];
		const pad = document.getElementById('pad');
		function rec(arr, e){ const r = pad.getBoundingClientRect();
			arr.push([Math.round(e.clientX - r.left), Math.round(e.clientY - r.top)]); }
		pad.addEventListener('pointerdown', e => { rec(window.__downs, e); rec(window.__points, e); });
		pad.addEventListener('pointermove', e => { if (e.buttons) rec(window.__points, e); });
	</script>
</body></html>
"""


@pytest.fixture(scope='module')
def canvas_server():
	server = HTTPServer()
	server.start()
	server.expect_request('/pad').respond_with_data(CANVAS_HTML, content_type='text/html')
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


async def _read(cdp, var):
	return json.loads(
		(
			await cdp.cdp_client.send.Runtime.evaluate(
				params={'expression': f'JSON.stringify(window.{var})', 'returnByValue': True},
				session_id=cdp.session_id,
			)
		)['result']['value']
	)


async def test_place_pixels_strokes_runs_and_covers_cells(browser_session, canvas_server):
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/pad')

	tools = Tools()
	# 4x4 grid over canvas box left=40, top=20, 160x160 → 40px cells; cell (gx,gy) center = ((gx+0.5)*40, (gy+0.5)*40).
	# A 3-cell run in row 0 (x 0..2) plus a single cell at (0,2): 4 pixels, but only 2 strokes.
	result = await tools.act(
		action=tools.registry.create_action_model()(
			place_pixels=PlacePixelsAction(
				cells=[(0, 0), (1, 0), (2, 0), (0, 2)],
				grid_width=4,
				grid_height=4,
				area=(40, 20, 160, 160),
				delay_ms=0,
			)
		),
		browser_session=browser_session,
	)
	assert result.error is None, result.error
	assert 'Painted 4 pixels in 2 strokes' in result.extracted_content

	cdp = await browser_session.get_or_create_cdp_session()
	downs = await _read(cdp, '__downs')
	points = await _read(cdp, '__points')

	# Two contiguous runs → two pen-down strokes, not four clicks.
	assert len(downs) == 2, f'expected 2 strokes, got {downs}'

	# All four cell centers were covered by the strokes.
	covered = {tuple(p) for p in points}
	for ex, ey in [(20, 20), (60, 20), (100, 20), (20, 100)]:
		assert any(abs(cx - ex) <= 2 and abs(cy - ey) <= 2 for cx, cy in covered), f'cell center {ex},{ey} not painted'


async def test_place_pixels_rejects_too_many_cells(browser_session, canvas_server):
	await browser_session.navigate_to(f'http://{canvas_server.host}:{canvas_server.port}/pad')
	tools = Tools()
	result = await tools.act(
		action=tools.registry.create_action_model()(
			place_pixels=PlacePixelsAction(
				cells=[(0, 0)] * 1501,
				grid_width=4,
				grid_height=4,
				area=(40, 20, 160, 160),
				delay_ms=0,
			)
		),
		browser_session=browser_session,
	)
	assert result.error is not None
	assert 'cap is 1500' in result.error
