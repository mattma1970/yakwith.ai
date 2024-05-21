from __future__ import annotations
from typing import Optional, Any, Literal
from collections.abc import Iterator
import openai
from attr import define, field, Factory
from griptape.artifacts import TextArtifact
from griptape.utils import PromptStack, import_optional_dependency
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import HuggingFaceTokenizer


@define
class vLLMChatPromptDriver(BasePromptDriver):
    """
    Attributes:
        base_url: vLLM OpenAI compliant server URL. e.g. http://localhost:8000/v1
        api_key: Hugginface Hub API key if pulling models from HF Hub. This is ignored if models are already stored locally on the server.
        organization: Not Used
        client: An `openai.OpenAI` client.
        model: Huggingface model name if model to be cached on server, or path to model if model located elsewhere.
        tokenizer: HF tokenizer for model.
        user: A user id. Can be used to track requests by user.
        response_format: An optional OpenAi Chat Completion response format. Currently only supports `json_object` which will enable OpenAi's JSON mode.
        seed: An optional OpenAi Chat Completion seed.
        params: Additional model specific run parameters for the HF model.
    """

    base_url: Optional[str] = field(
        default=None, kw_only=True, metadata={"serializable": True}
    )
    api_key: Optional[str] = field(
        default=None, kw_only=True, metadata={"serializable": True}
    )
    organization: Optional[str] = field(
        default=None, kw_only=True, metadata={"serializable": True}
    )
    client: openai.OpenAI = field(
        default=Factory(
            lambda self: openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
            ),
            takes_self=True,
        )
    )
    model: str = field(kw_only=True, metadata={"serializable": True})
    tokenizer: HuggingFaceTokenizer = field(
        default=Factory(
            lambda self: HuggingFaceTokenizer(
                tokenizer=import_optional_dependency(
                    "transformers"
                ).AutoTokenizer.from_pretrained(self.model),
                max_output_tokens=self.max_tokens,
            ),
            takes_self=True,
        ),
        kw_only=True,
    )
    user: str = field(default="", kw_only=True, metadata={"serializable": True})
    response_format: Optional[Literal["json_object"]] = field(
        default=None, kw_only=True, metadata={"serializable": True}
    )
    seed: Optional[int] = field(
        default=None, kw_only=True, metadata={"serializable": True}
    )
    params: Optional[dict] = field(
        factory=dict, kw_only=True, metadata={"serializable": True}
    )

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        result = self.client.chat.completions.with_raw_response.create(
            **self._merged_params(prompt_stack)
        )

        parsed_result = result.parse()

        if len(parsed_result.choices) == 1:
            return TextArtifact(value=parsed_result.choices[0].message.content.strip())
        else:
            raise Exception(
                "Completion with more than one choice is not supported yet."
            )

    def try_stream(self, prompt_stack: PromptStack) -> Iterator[TextArtifact]:
        result = self.client.chat.completions.create(
            **self._merged_params(prompt_stack), stream=True
        )

        for chunk in result:
            if len(chunk.choices) == 1:
                delta = chunk.choices[0].delta
            else:
                raise Exception(
                    "Completion with more than one choice is not supported yet."
                )

            if delta.content is not None:
                delta_content = delta.content

                yield TextArtifact(value=delta_content)

    def token_count(self, prompt_stack: PromptStack) -> int:
        return self.tokenizer.count_tokens(self.prompt_stack_to_string(prompt_stack))

    def _prompt_stack_to_messages(
        self, prompt_stack: PromptStack
    ) -> list[dict[str, Any]]:
        return [
            {"role": self.__to_openai_role(i), "content": i.content}
            for i in prompt_stack.inputs
        ]

    def _merged_params(self, prompt_stack: PromptStack) -> dict:
        core_params = {
            "model": self.model,
            "temperature": self.temperature,
            "stop": self.tokenizer.stop_sequences,
            "user": self.user,
            "seed": self.seed,
        }

        if self.response_format == "json_object":
            core_params["response_format"] = {"type": "json_object"}
            # JSON mode still requires a system input instructing the LLM to output JSON.
            prompt_stack.add_system_input(
                "Provide your response as a valid JSON object."
            )

        messages = self._prompt_stack_to_messages(prompt_stack)

        if self.max_tokens is not None:
            core_params["max_tokens"] = self.max_tokens

        core_params["messages"] = messages

        # Merge additional parameters and drop duplicates.
        """ for k,v in self.params.items():
            if not(k in core_params):
                core_params[k] = v """

        return core_params

    def __to_openai_role(self, prompt_input: PromptStack.Input) -> str:
        if prompt_input.is_system():
            return "system"
        elif prompt_input.is_assistant():
            return "assistant"
        else:
            return "user"
