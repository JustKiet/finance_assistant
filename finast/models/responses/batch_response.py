from pydantic import BaseModel
from openai.types.responses import Response
from openai.types.chat import ParsedChatCompletion
from typing import Union

class BatchResponse(BaseModel):
    index: int
    content: Union[Response, ParsedChatCompletion]
    cost: float
