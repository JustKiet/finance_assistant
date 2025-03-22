from abc import ABC, abstractmethod
from typing import Literal
from finast.models.responses import ParserResponse

class BaseParser(ABC):
    @abstractmethod
    async def async_parse_pdf(self, 
                              pdf_path: str,
                              batch_size: int,
                              mode: Literal["tabular", "html", "markdown"]) -> ParserResponse:
        raise NotImplementedError