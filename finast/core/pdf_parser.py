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

from finast.models.responses import ParserResponse, PageResponse, ImageObject
from finast.models.FLC.parent_finstate import SeperateCashFlowStatement
from finast.utils import CostTracker, ImageAugmentor

from loguru import logger

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
    
    def _construct_png(self, 
                      image: Image.Image) -> bytes:
        """Construct a valid PNG image from the PIL image."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def _batch_loader(self,
                      images: list[Image.Image],
                      batch_size: int = 3) -> list[Image.Image]:
        """Load a batch of images into memory."""
        
        batches = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batches.append(batch)

        return batches
    
    def _parse_client_response(self,
                               base64_images: list[str]) -> ParsedChatCompletion[SeperateCashFlowStatement]:
        """Request the OpenAI API to parse image into a structured format."""
        image_objects = [ImageObject(base64_image=image).image_url_dict for image in base64_images]

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
                "content": image_objects
            }
        ]
        
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=input_payload,
            response_format=SeperateCashFlowStatement
        )

        with open("misc_output/parsed_response.json", "w") as f:
            json.dump(response.model_dump(), f, indent=4, ensure_ascii=False)

        return response
    
    def _create_client_response(self,
                                base64_images: list[str],
                                mode: Literal["html", "markdown"] = "html") -> Response:
        """Request the OpenAI API to convert the image into markdown format."""
        # Construct the prompt based on the mode
        if mode == "markdown":
            mode_prompt = "Please convert this image into markdown format. For tables, use the html format."
        elif mode == "html":
            mode_prompt = "Please convert this image into HTML format."

        image_objects = [ImageObject(base64_image=image).input_image_dict for image in base64_images]

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
                "content": image_objects
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
                                 images: list[Image.Image],
                                 mode: Literal["html", "markdown", "tabular"] = "tabular"
                                 ) -> PageResponse:
        """Convert an image into markdown format."""
        
        processed_images = []

        # Image preprocessing logic (rotate and enhance)
        for image in images:
            image = ImageAugmentor.rotate_image(
                image=image,
                method="hough"
            )

            # Enhance the image
            image = ImageAugmentor.enhance_image(image=image)
            
            # Convert the image into a PNG
            image_png = self._construct_png(image=image)

            # ===============================
            # NOTE: For debugging purposes
            with open(f"output/{idx}.png", "wb") as f:
                f.write(image_png)
            # ===============================

            # Encode the image into a base64 string
            base64_image = self._encode_image(image=image_png)

            processed_images.append(base64_image)

        # Create a client response
        if mode == "html" or mode == "markdown":
            response = self._create_client_response(
                base64_images=processed_images,
                mode=mode
            )
        elif mode == "tabular":
            response = self._parse_client_response(
                base64_images=processed_images
            )

        # NOTE: Comment out the following code to disable mock response
        # Create a mock response object
        #with open("misc_output/parsed_response.json", "r") as f:
        #    response = ParsedChatCompletion(**json.load(f))

        page_response = self._handle_client_response(
            idx=idx,
            response=response,
        )

        return page_response
    
    def _handle_responses(self,
                          responses: list[PageResponse],
                          mode: Literal["html", "markdown", "tabular"] = "tabular") -> ParserResponse:
        """Handle the responses from the OpenAI API."""
                    # Calculate the total cost of the API calls
        total_cost = sum(response.cost for response in responses)
        
        # Calculate the total number of pages
        total_pages = len(responses)
        
        if isinstance(responses[0].content, ParsedChatCompletion):
            logger.info("Handling ParsedChatCompletion responses.")
            # Calculate the total number of input tokens
            total_input_tokens = sum(response.content.usage.prompt_tokens for response in responses)
            # Calculate the total number of output tokens
            total_output_tokens = sum(response.content.usage.completion_tokens for response in responses)
        
        elif isinstance(responses[0].content, Response):
            logger.info("Handling Response responses.")
            # Calculate the total number of input tokens
            total_input_tokens = sum(response.usage.input_tokens for response in responses)
            # Calculate the total number of output tokens
            total_output_tokens = sum(response.usage.output_tokens for response in responses)
        
        if mode == "html" or mode == "markdown":
            outputs = []

            for response in responses:
                outputs.append(response.content.output_text)
        
        elif mode == "tabular":
            outputs = []

            for response in responses:
                outputs.append(response.content.choices[0].message.parsed)

        logger.success(f"Parsing completed. Total cost: {total_cost}.")

        return ParserResponse(
            content=outputs,
            total_cost=total_cost,
            total_pages=total_pages,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )
    
    async def async_parse_pdf(self, 
                              pdf_path: str,
                              batch_size: int = 3,
                              mode: Literal["html", "markdown", "tabular"] = "tabular") -> ParserResponse:
        """Parse the PDF file into the desired format."""
        if mode:
            self.mode = mode

        # Convert all pages of the pdf into images
        images = convert_from_path(pdf_path=pdf_path)

        # Load the images into batches
        batches = self._batch_loader(
            images=images,
            batch_size=batch_size
        )

        # Convert the images into markdown format in parallel
        tasks = [
            self._async_image_parser(
                idx=idx,
                images=image,
                mode=mode,
            )
            for idx, image in enumerate(batches)
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
            responses=responses,
            mode=mode
        )