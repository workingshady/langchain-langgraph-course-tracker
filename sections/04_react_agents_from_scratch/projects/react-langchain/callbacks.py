from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        prompt = prompts[0] if prompts else ""
        print("\n========== LLM PROMPT ==========")
        print(prompt)
        print("=" * 32)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        gen = response.generations
        output_text = ""
        if gen and gen[0]:
            output_text = getattr(gen[0][0], 'text', '')
        print("\n========== LLM RESPONSE ==========")
        print(output_text)
        print("=" * 34)
