from pydantic import BaseModel
from openai.types.responses import Response

class PageResponse(BaseModel):
    page: int
    content: Response
    cost: float
