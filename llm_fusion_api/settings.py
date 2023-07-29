from starlette.config import Config
from starlette.datastructures import Secret


# Load environment variables from .env file
config = Config(".env")

## Settings
# Debug mode
DEBUG = config('DEBUG', cast=bool, default=False)
# Secret token for authentication
SECRET_TOKEN = config('SECRET_TOKEN', cast=Secret, default='')
# OpenAI API settings
OPENAI_API_BASE = config('OPENAI_API_BASE', default='https://api.openai.com/v1')
OPENAI_API_KEY = config('OPENAI_API_KEY', cast=Secret, default='')
# Wenxin API settings
WENXIN_API_KEY = config('WENXIN_API_KEY', cast=Secret, default='')
WENXIN_SECRET_KEY = config('WENXIN_SECRET_KEY', cast=Secret, default='')
# FastChat API settings
FASTCHAT_OPENAI_API_BASE = config('FASTCHAT_OPENAI_API_BASE', default='')
FASTCHAT_OPENAI_API_KEY = config('FASTCHAT_OPENAI_API_KEY', cast=Secret, default='')
