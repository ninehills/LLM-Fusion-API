from starlette.responses import JSONResponse

class ErrorResponse(JSONResponse):
    """Error response with OpenAI API format"""
    def __init__(self, status_code: int, message: str):
        content = {'error': {'message': message}}
        super().__init__(content, status_code=status_code)
