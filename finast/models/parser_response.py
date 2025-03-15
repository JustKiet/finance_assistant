from pydantic import BaseModel
from .page_response import PageResponse

class ParserResponse(BaseModel):
    total_pages: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    markdown: str
