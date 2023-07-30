from starlette.config import Config
from starlette.datastructures import Secret


# Load environment variables from .env file
config: Config = Config(".env")

## Settings
# Debug mode
DEBUG: bool = config('DEBUG', cast=bool, default=False)
# Secret token for authentication
SECRET_TOKEN: Secret = config('SECRET_TOKEN', cast=Secret, default=Secret(''))
# OpenAI API settings
OPENAI_API_BASE: str = config('OPENAI_API_BASE', default='https://api.openai.com/v1')
OPENAI_API_KEY: Secret = config('OPENAI_API_KEY', cast=Secret, default=Secret(''))
# Wenxin API settings
WENXIN_API_KEY: Secret = config('WENXIN_API_KEY', cast=Secret, default=Secret(''))
WENXIN_SECRET_KEY: Secret = config('WENXIN_SECRET_KEY', cast=Secret, default=Secret(''))
# FastChat API settings
FASTCHAT_OPENAI_API_BASE: str = config('FASTCHAT_OPENAI_API_BASE', default='')
FASTCHAT_OPENAI_API_KEY: Secret = config('FASTCHAT_OPENAI_API_KEY', cast=Secret, default=Secret(''))
# MiniMax API settings
MINIMAX_GROUP_ID: Secret = config('MINIMAX_GROUP_ID', cast=Secret, default=Secret(''))
MINIMAX_API_KEY: Secret = config('MINIMAX_API_KEY', cast=Secret, default=Secret(''))
