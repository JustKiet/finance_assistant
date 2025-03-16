import asyncio
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import os
import base64
import mimetypes
from pdf2image import convert_from_path
import io
from PIL import Image
from typing import Literal, Union
import numpy as np
import json

from openai.types.responses import Response
from openai.types.chat import ParsedChatCompletion
from openai._types import NOT_GIVEN

from finast.models.responses import ParserResponse, PageResponse
from finast.models.FLC.parent_finstate import CashFlowCumulativeUnit, CashFlowData, CashFlowBulletPoint, CashFlowSubSection, CashFlowSection, SeperateCashFlowStatement
from finast.utils import CostTracker, ImageAugmentor


load_dotenv()

client = OpenAI()

class OpenAIPDFParser:
    def __init__(self, 
                 client: OpenAI,
                 mode: Literal["html", "markdown", "tabular"] = "tabular") -> None:
        self.client = client
        self.mode = mode

    def _validate_image_path(self, 
                             image_path: str) -> str:
        """Validate that the file is a JPG, JPEG, or PNG."""
        valid_mime_types = {"image/jpeg", "image/png"}
        file_type = mimetypes.guess_type(image_path)[0]
        
        if file_type not in valid_mime_types:
            raise ValueError(f"Invalid file type: {file_type}. Only JPG, JPEG, and PNG are allowed.")
        
        return file_type

    def _encode_image(self, 
                      image: Image.Image) -> str:
        """Encode the image into a base64 string."""
        encoded_string = base64.b64encode(image).decode('utf-8')
        return encoded_string
    
    def construct_png(self, 
                      image: Image.Image) -> bytes:
        """Construct a valid PNG image from the PIL image."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def _parse_client_response(self,
                               base64_image: str) -> ParsedChatCompletion:

        input_payload = [
            {
                "role": "system",
                "content": f"""
                    You are a professional senior accountant. 
                    Please extract ONLY the table data in this image.
                    You MUST preserve the table structure (if any) in the output and all its values.
                    If the value in the table is wrapped in parentheses, it is negative. 
                    If you don't, the company will lose money or even have to face legal consequences.
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=input_payload,
            response_format=SeperateCashFlowStatement
        )

        return response
    
    def _create_client_response(self,
                                base64_image: str,
                                mode: Literal["html", "markdown"] = "html") -> Response:
        # Construct the prompt based on the mode
        if self.mode == "markdown":
            mode_prompt = "Please convert this image into markdown format. For tables, use the html format."
        elif self.mode == "html":
            mode_prompt = "Please convert this image into HTML format."

        # Construct the input payload before sending to OpenAI API
        input_payload = [
            {
                "role": "system",
                "content": f"""
                    You are a professional senior accountant. 
                    {mode_prompt}
                    You MUST preserve the table structure (if any) in the output and all its values.
                    If the value in the table is wrapped in parentheses, it is negative. 
                    If you don't, the company will lose money or even have to face legal consequences.
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]

        # Send the input payload to the OpenAI API
        response = self.client.responses.create(
            model="gpt-4o",
            input=input_payload,
        )

        return response
    
    def _handle_client_response(self,
                                idx: int,
                                response: Union[Response, ParsedChatCompletion]) -> PageResponse:
        """A handle function that parse the OpenAI client response and return a PageResponse object."""
        if isinstance(response, Response):
            # Track the cost of the API call
            cost = CostTracker.track_cost(
                input_tokens_count=response.usage.input_tokens,
                output_tokens_count=response.usage.output_tokens,
                model_name="gpt-4o",
            )

            page_response = PageResponse(
                page=idx,
                content=response,
                cost=cost
            )

            return page_response
        
        elif isinstance(response, ParsedChatCompletion):
            cost = CostTracker.track_cost(
                input_tokens_count=response.usage.prompt_tokens,
                output_tokens_count=response.usage.completion_tokens,
                model_name="gpt-4o"
            )

            page_response = PageResponse(
                page=idx,
                content=response,
                cost=cost
            )

            return page_response
        
    async def _async_image_parser(self, 
                                 idx: int,
                                 image: Image.Image,
                                 ) -> PageResponse:
        """Convert an image into markdown format."""
        # Preprocess the image (rotate and enhance)
        image = ImageAugmentor.rotate_image(
            image=image,
            method="hough"
        )
        
        # Convert the image into a PNG
        image_png = self.construct_png(image)

        # Encode the image into a base64 string
        base64_image = self._encode_image(image_png)

        # Create a client response
        if self.mode == "html" or self.mode == "markdown":
            response = self._create_client_response(
                base64_image=base64_image,
                mode=self.mode
            )
        elif self.mode == "tabular":
            response = self._parse_client_response(
                base64_image=base64_image
            )

        # NOTE: Comment out the following code to disable mock response
        # Create a mock response object
        #with open("output.json", "r") as f:
        #    response = Response(**json.load(f)[0])

        page_response = self._handle_client_response(
            idx=idx,
            response=response,
        )

        return page_response
    
    def _handle_responses(self,
                               responses: list[PageResponse]) -> ParserResponse:
        if self.mode == "html" or self.mode == "markdown":
            # Calculate the total cost of the API calls
            total_cost = sum(response.cost for response in responses)

            # Calculate the total number of pages
            total_pages = len(responses)

            # Calculate the total number of input tokens
            total_input_tokens = sum(response.content.usage.input_tokens for response in responses)

            # Calculate the total number of output tokens
            total_output_tokens = sum(response.content.usage.output_tokens for response in responses)

            outputs = []

            for response in responses:
                outputs.append(response.content.output_text)
            
            return ParserResponse(
                content=outputs,
                total_cost=total_cost,
                total_pages=total_pages,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )
        
        elif self.mode == "tabular":
            # Calculate the total cost of the API calls
            total_cost = sum(response.cost for response in responses)

            # Calculate the total number of pages
            total_pages = len(responses)

            # Calculate the total number of input tokens
            total_input_tokens = sum(response.content.usage.prompt_tokens for response in responses)

            # Calculate the total number of output tokens
            total_output_tokens = sum(response.content.usage.completion_tokens for response in responses)

            outputs = []

            for response in responses:
                outputs.append(response.content.choices[0].message.parsed)

            return ParserResponse(
                content=outputs,
                total_cost=total_cost,
                total_pages=total_pages,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )
    
    async def async_parse_pdf(self, 
                              pdf_path: str) -> ParserResponse:
        """Parse a pdf into markdown format."""
        # Convert all pages of the pdf into images
        images = convert_from_path(pdf_path=pdf_path)

        # Convert the images into markdown format in parallel
        tasks = [
            self._async_image_parser(
                idx=idx,
                image=image,
            )
            for idx, image in enumerate(images)
        ]
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks)

        # Sort the responses by the original page order
        responses = sorted(responses, key=lambda x: x.page)

        # ========================
        #json_responses = [response.model_dump() for _, response in responses]
        ## Save the responses to a JSON file
        #with open("output.json", "w") as f:
        #    json.dump(json_responses, f, indent=4)
        # ========================

        return self._handle_responses(
            responses=responses
        )