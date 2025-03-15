import asyncio
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import os
import base64
import mimetypes
from pdf2image import convert_from_path
import io
from PIL import Image
from pprint import pprint
import json
import cv2
import numpy as np

from openai.types.responses import Response

from finast.models import ParserResponse, PageResponse
from finast.utils import CostTracker, ImageAugmentor

load_dotenv()

client = OpenAI()

class PDFParser:
    def __init__(self, 
                 client: OpenAI):
        self.client = client

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

    async def _async_image_to_markdown(self, 
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

        # Construct the input payload before sending to OpenAI API
        input_payload = [
            {
                "role": "system",
                "content": """
                    You are a professional senior accountant. 
                    Please reformat this image into markdown format. 
                    For table structures, please use html format in order to preserve the complete table structure. 
                    You MUST preserve the table structure in the output and all its values. 
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

        # NOTE: Uncomment the following code to send the input payload to the OpenAI API
        # Send the input payload to the OpenAI API
        response = self.client.responses.create(
            model="gpt-4o",
            input=input_payload,
        )


        # NOTE: Comment out the following code to disable mock response
        # Create a mock response object
        #with open("output.json", "r") as f:
        #    response = Response(**json.load(f)[0])

        # Track the cost of the API call
        cost = CostTracker.track_cost(
            input_tokens_count=response.usage.input_tokens,
            output_tokens_count=response.usage.output_tokens,
            model_name="gpt-4o",
        )

        page_response = PageResponse(
            page=idx,
            content=response,
            cost=cost,
        )

        return page_response
    
    async def async_parse_pdf(self, 
                              pdf_path: str) -> ParserResponse:
        """Parse a pdf into markdown format."""
        # Convert all pages of the pdf into images
        images = convert_from_path(pdf_path=pdf_path)
        
        markdown_text = ""

        # Convert the images into markdown format in parallel
        tasks = [
            self._async_image_to_markdown(
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

        # Combine all the markdown text into a single string
        for response in responses:
            markdown_text += response.content.output_text + "\n\n"

        # Calculate the total cost of the API calls
        total_cost = sum(response.cost for response in responses)

        # Calculate the total number of pages
        total_pages = len(responses)

        # Calculate the total number of input tokens
        total_input_tokens = sum(response.content.usage.input_tokens for response in responses)

        # Calculate the total number of output tokens
        total_output_tokens = sum(response.content.usage.output_tokens for response in responses)

        parser_response = ParserResponse(
            markdown=markdown_text,
            total_cost=total_cost,
            total_pages=total_pages,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

        return parser_response