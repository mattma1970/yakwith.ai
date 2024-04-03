from typing import List, Dict, Any
from attr import define, Factory, field
from griptape.memory.structure import ConversationMemory
from griptape.memory.structure import Run
from voice_chat.utils.text_processing import remove_strings


@define
class PreprocConversationMemory(ConversationMemory):
    """Subclass of griptape ConversationMemory that adds pre-processing to conversation memory"""

    stop_sequences: List[str] = field(default=Factory(list), kw_only=True)

    def try_add_run(self, run: Run) -> None:
        processed_text: str = run.output
        if len(self.stop_sequences) > 0:
            for seq in self.stop_sequences:
                processed_text = processed_text.replace(seq, "")
        run.output = processed_text
        self.runs.append(run)

        if self.max_runs:
            while len(self.runs) > self.max_runs:
                self.runs.pop(0)
