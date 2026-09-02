import asyncio
import base64
import binascii
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel, ValidationError

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage, ContentPartImageParam
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOpenClaw(BaseChatModel):
	"""Use an OpenClaw raw model transport, including subscription-backed providers."""

	model: str = 'deepseek/deepseek-v4-pro'
	profile: str = 'gpt'
	openclaw_binary: str = 'openclaw'
	thinking: str | None = None

	@property
	def provider(self) -> str:
		return 'openclaw'

	@property
	def name(self) -> str:
		return self.model

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		prompt = self._format_prompt(messages, output_format)
		# Images can't ride inline in the text prompt; openclaw takes them as --file paths.
		# Decode every image content part to a temp PNG and clean them up after the call.
		image_paths = self._write_image_files(messages)
		command = [
			self.openclaw_binary,
			'--profile',
			self.profile,
			'infer',
			'model',
			'run',
			'--local',
			'--json',
			'--model',
			self.model,
			'--prompt',
			prompt,
		]
		for image_path in image_paths:
			command.extend(['--file', image_path])
		if self.thinking:
			command.extend(['--thinking', self.thinking])

		try:
			process = await asyncio.create_subprocess_exec(
				*command,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=self._get_environment(),
			)
			stdout, stderr = await process.communicate()
		finally:
			for image_path in image_paths:
				try:
					os.unlink(image_path)
				except OSError:
					pass
		if process.returncode != 0:
			error = stderr.decode().strip() or stdout.decode().strip() or 'OpenClaw model invocation failed'
			raise ModelProviderError(message=error, model=self.model)

		try:
			payload = json.loads(stdout)
			text = payload['outputs'][0]['text']
		except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
			raise ModelProviderError(message=f'Invalid OpenClaw model response: {e}', model=self.model) from e

		if output_format is None:
			return ChatInvokeCompletion(completion=text, usage=None)

		try:
			return ChatInvokeCompletion(completion=output_format.model_validate_json(self._unwrap_json_fence(text)), usage=None)
		except ValidationError as e:
			raise ModelProviderError(message=f'Invalid structured OpenClaw model response: {e}', model=self.model) from e

	@staticmethod
	def _unwrap_json_fence(text: str) -> str:
		"""Remove an optional Markdown fence around a structured JSON response."""
		match = re.fullmatch(r'\s*```(?:json)?\s*(.*?)\s*```\s*', text, flags=re.DOTALL | re.IGNORECASE)
		return match.group(1) if match else text

	def _get_environment(self) -> dict[str, str]:
		environment = os.environ.copy()
		state_directory = Path.home() / f'.openclaw-{self.profile}'
		environment['OPENCLAW_STATE_DIR'] = str(state_directory)
		environment['OPENCLAW_CONFIG_PATH'] = str(state_directory / 'openclaw.json')
		return environment

	@staticmethod
	def _write_image_files(messages: list[BaseMessage]) -> list[str]:
		"""Decode base64 data-URL images from message content into temp files for openclaw --file.

		Only data: URLs are materialized; http(s) image URLs are skipped (openclaw --file expects a
		local path). Returns the list of temp file paths in message/content order.
		"""
		paths: list[str] = []
		for message in messages:
			content = getattr(message, 'content', None)
			if not isinstance(content, list):
				continue
			for part in content:
				if not isinstance(part, ContentPartImageParam):
					continue
				url = part.image_url.url
				if not url.startswith('data:'):
					continue
				try:
					header, b64 = url.split(',', 1)
					raw = base64.b64decode(b64)
				except (ValueError, binascii.Error):
					continue
				suffix = '.png'
				if 'image/jpeg' in header or 'image/jpg' in header:
					suffix = '.jpg'
				elif 'image/webp' in header:
					suffix = '.webp'
				fd, path = tempfile.mkstemp(prefix='openclaw-img-', suffix=suffix)
				with os.fdopen(fd, 'wb') as f:
					f.write(raw)
				paths.append(path)
		return paths

	def _format_prompt(self, messages: list[BaseMessage], output_format: type[BaseModel] | None) -> str:
		parts = [f'<{message.role}>\n{message.text}\n</{message.role}>' for message in messages]
		if output_format is not None:
			schema = json.dumps(output_format.model_json_schema(), indent=2)
			parts.append(
				'<output_requirements>\n'
				'Return only valid JSON matching this schema. Do not use Markdown fences or add commentary.\n'
				f'{schema}\n'
				'</output_requirements>'
			)
		return '\n\n'.join(parts)
