from attr import define, field, Factory
from typing import Callable, Dict, List, Union
from griptape.utils import PromptStack
from griptape.tokenizers import HuggingFaceTokenizer
from griptape.events import FinishStructureRunEvent
from transformers import AutoTokenizer
import logging

logger = logging.getLogger("YakChatAPI")


class PromptStackUtils:
    @classmethod
    def autotokenizer_prompt_stack_to_string(
        cls, hf_tokenizer: HuggingFaceTokenizer
    ) -> Callable[[PromptStack], str]:
        """
        Apply the Autotokenizer chat template to the griptape.ai prompt stack.
        @args:
            tokenizer: Autotokenizer : The hugging face tokenize with the apply_chat_template function defined in it.
        @returns:
            callable([PromptStack], data_field_name: str): a function that applies the tokenizer.apply_chat_template function.
                    note: the the full prompt may consist of several PromptStacks (e.g. current, history) hence the list[PromptStack]
        """

        def _apply_chat_template(
            template_func: Callable, data_field: str = "inputs", **kwargs: Dict
        ) -> Callable:
            def inner(prompt_stack: PromptStack) -> str:
                inputs = getattr(prompt_stack, data_field)
                return template_func(inputs, **kwargs)

            return inner

        def stack_to_string_func(stack: Union[PromptStack, List[PromptStack]]) -> str:
            if isinstance(stack, PromptStack):
                stack = [stack]
            stack_string: List[str] = list(
                map(
                    _apply_chat_template(
                        hf_tokenizer.tokenizer.apply_chat_template,
                        "inputs",
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                    stack,
                )
            )
            ret = "".join(stack_string)
            logger.debug(f"{__name__}: {ret}, length: {len(ret)}")
            return ret

        return stack_to_string_func


if __name__ == "__main__":
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            "/home/mtman/Documents/Repos/yakwith.ai/models/Mistral-7B-OpenOrca"
        ),
        max_output_tokens=1000,
    )
    stack: PromptStack = PromptStack()
    stack.add_system_input("The system input")
    stack.add_user_input("Hello from the user")
    stack.add_assistant_input("Hello from the assistant")

    all_stacks = [stack]
    stack_to_string = PromptStackUtils.autotokenizer_prompt_stack_to_string(tokenizer)
    result = stack_to_string(all_stacks)
    print(result)
