# Website Pet Chrome Extension

Load this folder as an unpacked Chrome extension. It injects a small task prompt
into normal web pages and sends tasks to the local browser-use bridge.

Chrome must have remote debugging enabled so browser-use can control the tab.

The prompt panel also displays live task status, opens automatically when the
agent asks a question, accepts replies, and can stop the active task.

The persistent on-page cartoon cat companion uses PixiJS frame animations from
Mattz Art's Cat 2D Pixel Art pack. Browser-use moves that same companion to the
real target before click and input actions.

For local animation checks, use `1` for idle, `2` for walk, `3` for jump, `4`
for sleep, `5` for lick, `6` for click, and `7` for success. The free pack does
not include sleep or lick sheets yet, so those two currently fall back to idle.
