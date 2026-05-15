"""Module for loading language models.

Depending on the configuration, different language models are loaded and
different libraries are used. The main goal is to provide a unified interface
to the different models and libraries.
"""

import abc
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, TypeGuard, cast

from typing_extensions import override

from gerd.models.model import ChatMessage, ChatRole, ModelConfig, ModelEndpoint

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from requests import Response


class LLM:
    """The abstract base class for large language models.

    Should be implemented by all language model backends.
    """

    @abc.abstractmethod
    def __init__(self, config: ModelConfig) -> None:
        """A language model is initialized with a configuration.

        Parameters:
            config: The configuration for the language model
        """
        pass

    @abc.abstractmethod
    def generate(self, prompt: str, config: ModelConfig | None = None) -> str:
        """Generate text based on a prompt.

        Parameters:
            prompt: The prompt to generate text from

        Returns:
            The generated text
        """
        pass

    @abc.abstractmethod
    def create_chat_completion(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> tuple[ChatRole, str]:
        """Create a chat completion based on a list of messages.

        Parameters:
            messages: The list of messages in the chat history

        Returns:
            The role of the generated message and the content
        """
        pass


class MockLLM(LLM):
    """A mock language model for testing purposes."""

    @override
    def __init__(self, _: ModelConfig) -> None:
        self.ret_value = "MockLLM"
        pass

    @override
    def generate(self, _unused: str, _not_used: ModelConfig | None = None) -> str:
        return self.ret_value

    @override
    def create_chat_completion(
        self, _unused: list[ChatMessage], _not_used: ModelConfig | None = None
    ) -> tuple[ChatRole, str]:
        return ("assistant", self.ret_value)


class LlamaCppLLM(LLM):
    """A language model using the Llama.cpp library."""

    @override
    def __init__(self, config: ModelConfig) -> None:
        from llama_cpp import Llama

        self.config = config
        self._model = Llama.from_pretrained(
            repo_id=config.name,
            filename=config.file,
            n_ctx=config.context_length,
            n_gpu_layers=config.gpu_layers,
            n_threads=config.threads,
            **config.extra_kwargs or {},
        )

    @override
    def generate(self, prompt: str, config: ModelConfig | None = None) -> str:
        config = config or self.config
        res = self._model(
            prompt,
            stop=config.stop,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            temperature=config.temperature,
            repeat_penalty=config.repetition_penalty,
        )
        output = next(res) if isinstance(res, Iterator) else res
        return output["choices"][0]["text"]

    @override
    def create_chat_completion(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> tuple[ChatRole, str]:
        config = config or self.config
        res = self._model.create_chat_completion(
            # mypy cannot resolve the role parameter even though
            # is is defined on compatible literals
            [{"role": m["role"], "content": m["content"]} for m in messages],  # type: ignore[misc]
            stop=config.stop,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            temperature=config.temperature,
            repeat_penalty=config.repetition_penalty,
        )
        if not isinstance(res, Iterator):
            msg = res["choices"][0]["message"]
            if msg["role"] == "function":
                error_msg = "function role not expected"
                raise NotImplementedError(error_msg)
            return (msg["role"], msg["content"].strip() if msg["content"] else "")

        error_msg = "Cannot process stream responses for now"
        raise NotImplementedError(error_msg)


class TransformerLLM(LLM):
    """A language model using the transformers library."""

    @override
    def __init__(self, config: ModelConfig) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedModel,
            pipeline,
        )

        # use_fast=False is ignored by transformers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.config = config
        torch_dtypes: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

        model_kwargs = dict(config.extra_kwargs or {})
        model_kwargs.pop("enable_thinking", None)
        model_kwargs.pop("chat_template_kwargs", None)
        if config.torch_dtype in torch_dtypes:
            model_kwargs["torch_dtype"] = torch_dtypes[config.torch_dtype]

        tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.name, **model_kwargs
        )

        loaded_loras = set()
        for lora in config.loras:
            _LOGGER.info("Loading adapter %s", lora)
            if not Path(lora / "adapter_model.safetensors").exists():
                _LOGGER.warning("Adapter %s does not exist", lora)
                continue
            model.load_adapter(str(lora))
            loaded_loras.add(lora)
            train_params = Path(lora) / "training_parameters.json"
            if train_params.exists() and tokenizer.pad_token_id is None:
                from gerd.training.lora import LoraTrainingConfig

                with open(train_params, "r") as f:
                    lora_config = LoraTrainingConfig.model_validate_json(f.read())
                    tokenizer.pad_token_id = lora_config.pad_token_id
                    # https://github.com/huggingface/transformers/issues/34842#issuecomment-2490994584
                    tokenizer.padding_side = (
                        "left" if lora_config.padding_side == "right" else "right"
                    )
                    # tokenizer.padding_side = lora_config.padding_side

        if loaded_loras:
            model.enable_adapters()

        self._pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            # device_map="auto",  # https://github.com/huggingface/transformers/issues/31922
            device=(
                "cuda"
                if config.gpu_layers > 0
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            ),
            use_fast=False,
        )

    @override
    def generate(self, prompt: str, config: ModelConfig | None = None) -> str:
        config = config or self.config
        res = self._pipe(
            prompt,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            do_sample=True,
        )
        output: str = res[0]["generated_text"]
        return output

    @override
    def create_chat_completion(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> tuple[ChatRole, str]:
        config = config or self.config
        if config.extra_kwargs and config.extra_kwargs.get("enable_thinking"):
            _LOGGER.warning(
                "enable_thinking is not supported by TransformerLLM, ignoring. "
                "Use RemoteLLM with a compatible server for thinking support."
            )
        msg = self._pipe(
            [{"role": m["role"], "content": m["content"]} for m in messages],
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            do_sample=True,
        )[0]["generated_text"][-1]
        role = msg["role"]
        if _is_valid_role(role):
            return (role, msg["content"].strip())
        error_msg = "Unknown role: %s" % role
        raise ValueError(error_msg)


class RemoteLLM(LLM):
    """A language model using a remote endpoint.

    The endpoint can be any service that are compatible with llama.cpp and openai API.
    For further information, please refer to the llama.cpp
    [server API](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).
    """

    @override
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        if config.endpoint is None:
            msg = "Endpoint is required for remote LLM"
            raise ValueError(msg)

    @override
    def generate(self, prompt: str, config: ModelConfig | None = None) -> str:
        import json

        import requests

        config = config or self.config
        if config.endpoint is None:
            msg = "Endpoint is required for remote LLM"
            raise ValueError(msg)

        headers = {"Content-Type": "application/json"}
        if config.endpoint.key:
            headers["Authorization"] = (
                f"Bearer {config.endpoint.key.get_secret_value()}"
            )

        if config.endpoint and config.endpoint.type != "llama.cpp":
            msg = (
                "Only llama.cpp supports simple completion yet. "
                "Use chat completion instead."
            )
            raise NotImplementedError(msg)

        req = {
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repeat_penalty": config.repetition_penalty,
            "n_predict": config.max_new_tokens,
            "stop": config.stop or [],
            "prompt": prompt,
        }

        res = requests.post(
            config.endpoint.url + "/completion",
            headers=headers,
            data=json.dumps(req),
            timeout=300,
        )
        if res.status_code == 200:
            return str(res.json()["content"])
        else:
            _LOGGER.warning("Server returned error code %d", res.status_code)
        return ""

    def _build_chat_completion_request(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Builds the request for chat completion based on the endpoint type.

        For OpenAI-compatible endpoints, the request follows the OpenAI API format.
        For llama.cpp endpoints, the request follows the llama.cpp server API format.

        Parameters:
            messages: The list of messages in the chat history
            config: The model configuration

        Returns:
            A tuple of headers and the request body
        """
        headers = {"Content-Type": "application/json"}
        if config.endpoint and config.endpoint.key:
            headers["Authorization"] = (
                f"Bearer {config.endpoint.key.get_secret_value()}"
            )

        if config.endpoint is None:
            msg = "Endpoint is required for remote LLM"
            raise ValueError(msg)

        if config.endpoint.type == "openai":
            req: dict[str, Any] = {
                "model": config.name,
                "temperature": config.temperature,
                "frequency_penalty": config.repetition_penalty,
                "max_completion_tokens": config.max_new_tokens,
                "n": 1,
                "stop": config.stop,
                "top_p": config.top_p,
            }
            self._apply_openai_extra_kwargs(req, config)
        else:
            msg = f"Endpoint type '{config.endpoint.type}' is not supported"
            raise NotImplementedError(msg)

        req["messages"] = messages
        return headers, req

    @staticmethod
    def _apply_openai_extra_kwargs(req: dict[str, Any], config: ModelConfig) -> None:
        """Applies extra kwargs for OpenAI-compatible endpoints.

        Some features like enable_thinking are only supported by
        OpenAI-compatible endpoints.

        Parameters:
            req: The request body to be sent to the endpoint
            config: The model configuration containing extra kwargs
        Returns:
            None
        """
        if not config.extra_kwargs:
            return

        for key in (
            "enable_thinking",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        ):
            if key in config.extra_kwargs:
                req[key] = config.extra_kwargs[key]

    def _parse_openai_chat_response(self, res: "Response") -> tuple[ChatRole, str]:
        """Parses the response from an OpenAI-compatible chat completion endpoint.

        Parameters:
            res: The response object from the requests library
        Returns:
            A tuple of the role and content of the parsed response
        """
        try:
            j = res.json()
        except Exception as e:  # pragma: no cover - defensive
            msg = "Invalid JSON response from model server"
            _LOGGER.exception(msg)
            raise ValueError(msg) from e

        if not isinstance(j, dict) or "choices" not in j:
            msg = "Model server returned unexpected response structure: " f"{res.text}"
            _LOGGER.error(msg)
            raise ValueError(msg)

        try:
            res_message: dict[str, str] = j["choices"][0]["message"]
        except Exception as e:  # pragma: no cover - defensive
            msg = "Model server response missing choices/message"
            _LOGGER.exception(msg)
            _LOGGER.error("Full response text: %s", res.text)
            raise ValueError(msg) from e

        if _is_valid_role(res_message.get("role", "")):
            content = (res_message.get("content") or "").strip()
            reasoning_content = res_message.get("reasoning_content")
            if reasoning_content:
                reasoning = str(reasoning_content).strip()
                if reasoning:
                    # Reuse the existing rag.py parser to surface reasoning in UI.
                    content = f"<think>{reasoning}</think>{content}"
            role = cast(ChatRole, res_message.get("role", "assistant"))
            return (role, content)

        msg = "Unknown role: %s" % res_message.get("role")
        _LOGGER.error(msg)
        raise ValueError(msg)

    @override
    def create_chat_completion(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> tuple[ChatRole, str]:
        """Creates a chat completion by sending a request to the remote endpoint.

        Parameters:
            messages: The list of messages in the chat history
            config: Optional model configuration to override the default one
        Returns:
            A tuple of the role and content of the generated message
        """
        import json

        import requests

        config = config or self.config
        if config.endpoint is None:
            msg = "Endpoint is required for remote LLM"
            raise ValueError(msg)
        headers, req = self._build_chat_completion_request(messages, config)
        res = requests.post(
            config.endpoint.url + "/v1/chat/completions",
            headers=headers,
            data=json.dumps(req),
            timeout=300,
        )
        if res.status_code == 200:
            return self._parse_openai_chat_response(res)
        else:
            _LOGGER.warning("Server returned error code %d", res.status_code)
        return ("assistant", "")


def _is_valid_role(role: str) -> TypeGuard[ChatRole]:
    """Checks if the role is a valid ChatRole.

    Parameters:
        role: The role to check
    Returns:
            True if the role is valid, False otherwise
    """
    return role in {"user", "assistant", "system"}


def load_model_from_config(config: ModelConfig) -> LLM:
    """Loads a language model based on the configuration.

    Which language model is loaded depends on the configuration.
    For instance, if an endpoint is provided, a remote language model is loaded.
    If a file is provided, Llama.cpp is used.
    Otherwise, transformers is used.

    Parameters:
        config: The configuration for the language model

    Returns:
        The loaded language model

    """
    if config.endpoint:
        _LOGGER.info("Using remote endpoint %s", config.endpoint.url)
        return RemoteLLM(config)
    if config.file:
        _LOGGER.info(
            "Using Llama.cpp with model %s and file %s", config.name, config.file
        )
        return LlamaCppLLM(config)
    _LOGGER.info("Using transformers with model %s", config.name)
    return TransformerLLM(config)
