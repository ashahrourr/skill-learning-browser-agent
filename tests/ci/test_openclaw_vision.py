"""Unit tests for ChatOpenClaw image handling.

Regression guard: ChatOpenClaw used to drop image content entirely (it serialized only
message.text), so vision-capable models behind the openclaw transport never saw screenshots.
Images are now decoded to temp files and passed via openclaw's --file option.
"""

import base64
import os

from browser_use.llm.messages import (
	ContentPartImageParam,
	ContentPartTextParam,
	ImageURL,
	UserMessage,
)
from browser_use.llm.openclaw.chat import ChatOpenClaw

# 1x1 red pixel PNG.
_PNG_BYTES = base64.b64decode(
	'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
)
_DATA_URL = 'data:image/png;base64,' + base64.b64encode(_PNG_BYTES).decode()


def _cleanup(paths):
	for p in paths:
		try:
			os.unlink(p)
		except OSError:
			pass


def test_data_url_image_is_written_to_temp_file():
	msg = UserMessage(
		content=[
			ContentPartTextParam(text='look at this'),
			ContentPartImageParam(image_url=ImageURL(url=_DATA_URL)),
		]
	)
	paths = ChatOpenClaw._write_image_files([msg])
	try:
		assert len(paths) == 1
		assert os.path.exists(paths[0])
		assert open(paths[0], 'rb').read() == _PNG_BYTES
	finally:
		_cleanup(paths)


def test_text_only_messages_produce_no_files():
	msg = UserMessage(content='just text')
	assert ChatOpenClaw._write_image_files([msg]) == []


def test_http_image_urls_are_skipped():
	# openclaw --file needs a local path; remote URLs can't be materialized here.
	msg = UserMessage(
		content=[ContentPartImageParam(image_url=ImageURL(url='https://example.com/x.png'))]
	)
	assert ChatOpenClaw._write_image_files([msg]) == []


def test_multiple_images_preserve_order():
	red = _DATA_URL
	blue_bytes = base64.b64decode(
		'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwAEhQGAVqdkRwAAAABJRU5ErkJggg=='
	)
	blue = 'data:image/png;base64,' + base64.b64encode(blue_bytes).decode()
	msg = UserMessage(
		content=[
			ContentPartImageParam(image_url=ImageURL(url=red)),
			ContentPartImageParam(image_url=ImageURL(url=blue)),
		]
	)
	paths = ChatOpenClaw._write_image_files([msg])
	try:
		assert len(paths) == 2
		assert open(paths[0], 'rb').read() == _PNG_BYTES
		assert open(paths[1], 'rb').read() == blue_bytes
	finally:
		_cleanup(paths)


def test_jpeg_data_url_gets_jpg_suffix():
	jpeg = 'data:image/jpeg;base64,' + base64.b64encode(b'\xff\xd8\xff\xe0fake').decode()
	msg = UserMessage(content=[ContentPartImageParam(image_url=ImageURL(url=jpeg))])
	paths = ChatOpenClaw._write_image_files([msg])
	try:
		assert paths[0].endswith('.jpg')
	finally:
		_cleanup(paths)
