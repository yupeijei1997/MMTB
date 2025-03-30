from .toolace_handle import ToolACEMultiTurnMessages
from .xlam_handle import XLAMMultiTurnMessages
from .gorilla_handle import GorillaMultiTurnMessages
from .gpt_handle import GPTMultiTurnMessages, GPTAZUREMultiTurnMessages
from .llama_handle import LlamaMultiTurnMessages
from .qwen_handle import QwenMultiTurnMessages
from .chatglm_handle import ChatGLMMultiTurnMessages
from .hammer_handle import HammerMultiTurnMessages
from .watt_handle import WattMultiTurnMessages
from .fcm_handle import FCMMultiTurnMessages


tool_handle_map = {
    "toolace": (ToolACEMultiTurnMessages, False),
    "xlam": (XLAMMultiTurnMessages ,False),
    "gorilla": (GorillaMultiTurnMessages ,False),
    "chatglm": (ChatGLMMultiTurnMessages, False),
    "gpt4o": (GPTAZUREMultiTurnMessages, True),
    "gemini": (GPTMultiTurnMessages, True),
    "claude": (GPTMultiTurnMessages, True),
    "mistral": (GPTMultiTurnMessages, True),
    "fcm3.1": (FCMMultiTurnMessages, True),
    # Watt
    "watt70b": (WattMultiTurnMessages, True),
    "watt8b": (WattMultiTurnMessages, True),
    # Hammer
    "hammer7b": (HammerMultiTurnMessages, False),
    "hammer3b": (HammerMultiTurnMessages, False),
    "hammer1.5b": (HammerMultiTurnMessages, False),
    "hammer0.5b": (HammerMultiTurnMessages, False),
    # LLAMA
    "llama70b": (LlamaMultiTurnMessages, True),
    "llama8b": (LlamaMultiTurnMessages, True),
    "llama3b": (LlamaMultiTurnMessages, True),
    "llama1b": (LlamaMultiTurnMessages, True),
    # QWEN
    "qwen72b": (QwenMultiTurnMessages, True),
    "qwen32b": (QwenMultiTurnMessages, True),
    "qwen14b": (QwenMultiTurnMessages, True),
    "qwen7b": (QwenMultiTurnMessages, True),
    "qwen3b": (QwenMultiTurnMessages, True),
    "qwen1.5b": (QwenMultiTurnMessages, True),
    "qwen0.5b": (QwenMultiTurnMessages, True)
}
