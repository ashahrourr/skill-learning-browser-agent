"""Real-browser test for the coordinate drag action (DragCoordinateEvent).

Verifies that dispatching a drag fires a connected press → move(s) → release sequence on a
<canvas>, so freehand strokes (pixel editors, signature pads) register a continuous path
rather than two disconnected clicks.
"""

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserSession
from browser_use.browser.events import DragCoordinateEvent
from browser_use.browser.profile import BrowserProfile, ViewportSize

# A canvas that records every pointer event so the test can assert on the stroke shape.
CANVAS_HTML = """
<html><body style="margin:0">
	<canvas id="pad" width="400" height="400" style="width:400px;height:400px;background:#000"></canvas>
	<script>
		window.__events = [];
		const pad = document.getElementById('pad');
		function rec(type, e) {
			const r = pad.getBoundingClientRect();
			window.__events.push({type, x: Math.round(e.clientX - r.left), y: Math.round(e.clientY - r.top)});
		}
		// CDP dispatches mouse events which the browser synthesizes into pointer events;
		// listen to pointer only so a single drag is recorded once, not twice.
		pad.addEventListener('pointerdown', e => rec('down', e));
		pad.addEventListener('pointermove', e => { if (e.buttons) rec('move', e); });
		pad.addEventListener('pointerup', e => rec('up', e));
	</script>
</body></html>
"""


@pytest.fixture(scope='module')
def canvas_server():
	server = HTTPServer()
	server.start()
	server.expect_request('/canvas').respond_with_data(CANVAS_HTML, content_type='text/html')
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


async def test_drag_paints_connected_stroke_on_canvas(browser_session, canvas_server):
	url = f'http://{canvas_server.host}:{canvas_server.port}/canvas'
	await browser_session.navigate_to(url)

	# Drag a diagonal line across the canvas.
	path = [(50, 50), (200, 200), (350, 350)]
	event = browser_session.event_bus.dispatch(DragCoordinateEvent(path=path, steps_between=6))
	await event
	result = await event.event_result(raise_if_any=True, raise_if_none=False)
	assert result['points'] == 3

	cdp = await browser_session.get_or_create_cdp_session()
	events = (
		await cdp.cdp_client.send.Runtime.evaluate(
			params={'expression': 'JSON.stringify(window.__events)', 'returnByValue': True},
			session_id=cdp.session_id,
		)
	)['result']['value']

	import json

	recorded = json.loads(events)
	types = [e['type'] for e in recorded]

	# Exactly one press and one release, with held moves in between = a single connected stroke.
	assert types.count('down') == 1, f'expected one pointerdown, got {types}'
	assert types.count('up') == 1, f'expected one pointerup, got {types}'
	assert types[0] == 'down' and types[-1] == 'up'
	moves = [e for e in recorded if e['type'] == 'move']
	assert len(moves) >= 3, f'expected interpolated moves between points, got {len(moves)}'

	# Stroke should progress from the start toward the end (monotonic-ish along the diagonal).
	assert recorded[0]['x'] <= 60 and recorded[0]['y'] <= 60
	assert recorded[-1]['x'] >= 340 and recorded[-1]['y'] >= 340
