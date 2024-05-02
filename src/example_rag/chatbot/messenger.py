import copy
import logging
from typing import Dict, List, Tuple

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from src.example_rag.chatbot.model import MiniLlama
from src.example_rag.settings import CONTEXT_PROMPT, SYSTEM_PROMPT

LOGGER = logging.getLogger(__name__)


class Messenger:
    def __init__(
        self, chat_model: MiniLlama, context_retriever: VectorIndexRetriever
    ) -> None:
        self.chat_model = chat_model
        self.context_retriever = context_retriever
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        return ''.join([f'\n\n"{node.text}"\n\n...' for node in nodes])

    def _format_display_context(self, nodes: List[NodeWithScore]) -> str:
        return '\n\n' + ''.join(
            [
                f'{node.metadata["file_name"]}:\n{node.text} \n\n...\n\n'
                for node in nodes
            ]
        )

    def _prepare_messages(self) -> List[Dict[str, str]]:
        messages = copy.deepcopy(self.messages)
        messages[-1]['content'] = CONTEXT_PROMPT.format(
            context_str=messages[-1]['context'], query_str=messages[-1]['content']
        )
        messages = [
            {'role': message['role'], 'content': message['content']}
            for message in messages
        ]
        return messages

    def _process_input(self, content: str) -> None:
        nodes = self.context_retriever.retrieve(content)
        self.messages.append(
            {'role': 'user', 'content': content, 'context': self._format_context(nodes)}
        )
        return nodes

    def _get_response(self) -> str:
        messages = self._prepare_messages()
        response = self.chat_model.generate_response(messages)
        self.messages.append({'role': 'system', 'content': response})
        return response

    def chat(self, content: str, return_nodes: bool = False) -> Tuple[str, List[NodeWithScore]]:
        logging.info('Message received, processing...')
        nodes = self._process_input(content)
        if return_nodes:
            return self._get_response(), nodes
        else:
            return self._get_response(), self._format_display_context(nodes)

    def reset_context(self) -> None:
        logging.info('Resetting context...')
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
