from pydantic import BaseModel
from .page_response import PageResponse
from ..FLC.parent_finstate import SeperateCashFlowStatement
from typing import Union, List

class ParserResponse(BaseModel):
    total_pages: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    content: Union[List[str], List[SeperateCashFlowStatement]]
