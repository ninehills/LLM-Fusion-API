from starlette.config import Config
from starlette.datastructures import Secret


# Load environment variables from .env file
config = Config(".env")

## Settings
# Debug mode
DEBUG = config('DEBUG', cast=bool, default=False)
# Secret token for authentication
SECRET_TOKEN = config('SECRET_TOKEN', cast=Secret, default=None)

OPENAI_API_BASE = config('OPENAI_API_BASE', default='https://api.openai.com/v1')
OPENAI_API_KEY = config('OPENAI_API_KEY', cast=Secret, default=None)
