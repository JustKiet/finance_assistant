import asyncio
from openai import OpenAI
import base64
import mimetypes
from pdf2image import convert_from_path
import io
from PIL import Image
from typing import Literal, Union
import json

from openai.types.responses import Response
from openai.types.chat import ParsedChatCompletion
from openai._types import NOT_GIVEN

from finast.models.responses import ParserResponse, BatchResponse, ImageObject
from finast.models.FLC.finstate import SeperateCashFlowStatement
from finast.utils import CostTracker, ImageProcessor
from finast.interfaces import BaseParser

from loguru import logger

class OpenAIPDFParser(BaseParser):
    """
    A class that handles the PDF parsing using the OpenAI API.

    Args:
        client (OpenAI): The OpenAI client object.
        mode (Literal["html", "markdown", "tabular"]): The desired output

    ## Pipeline:
    1. Convert the PDF into images.
    2. Load the images into batches.
    3. Preprocess the images (eg: rotate and enhance).
    3. Convert the images into base64 strings.
    4. Send the images to the OpenAI API for parsing.
    5. Handle the responses and calculate the cost.
    6. Return the parsed response object.

    Examples:
    ```python
    import asyncio
    from openai import OpenAI
    from dotenv import load_dotenv
    from finast.core import OpenAIPDFParser
    import json

    load_dotenv()

    client = OpenAI()
        
    async def main():
        pdf_parser = OpenAIPDFParser(
            client=client,
            mode="tabular"
        )

        output = await pdf_parser.async_parse_pdf(
            pdf_path="docs/FLC_Baocaotaichinh_Q4_2020_Hopnhat_cashflow.pdf"
        )
        
        with open("output/cashflow_2020.json", "w") as f:
            json.dump(output.model_dump(), f, indent=4, ensure_ascii=False)

    if __name__ == "__main__":
        asyncio.run(main())
    """
    def __init__(self, 
                 client: OpenAI,
                 mode: Literal["html", "markdown", "tabular"] = "tabular") -> None:
        self.client = client
        self.mode = mode

    def _validate_image_path(self, 
                             image_path: str) -> Literal["image/jpeg", "image/png"]:
        """
        Validate that the file is a JPG, JPEG, or PNG.

        Args:
            image_path (str): The path to the image file.

        Returns:
            Literal["image/jpeg", "image/png"]: The mime type of the image.
        """
        valid_mime_types = {"image/jpeg", "image/png"}
        file_type = mimetypes.guess_type(image_path)[0]
        
        if file_type not in valid_mime_types:
            raise ValueError(f"Invalid file type: {file_type}. Only JPG, JPEG, and PNG are allowed.")
        
        return file_type

    def _encode_image(self, 
                      image: Image.Image) -> str:
        """
        Encode the image into a base64 string.

        Args:
            image (Image.Image): The PIL image object.

        Returns:
            str: The base64-encoded image string.
        """
        encoded_string = base64.b64encode(image).decode('utf-8')
        return encoded_string
    
    def _construct_png(self, 
                      image: Image.Image) -> bytes:
        """
        Construct a valid PNG image from the PIL image.

        Args:
            image (Image.Image): The PIL image object.
        
        Returns:
            bytes: The .PNG image bytes.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def _batch_loader(self,
                      images: list[Image.Image],
                      batch_size: int = 3) -> list[Image.Image]:
        """
        Load a batch of images into memory.

        Args:
            images (list[Image.Image]): The list of images.
            batch_size (int): The batch size for the image loader.

        Returns:
            list[Image.Image]: The list of image batches.
        """
        
        batches = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batches.append(batch)

        return batches
    
    # TODO: Generalize the function to accept multiple table formats! SeperateCashFlowStatement is just one of them.
    def _parse_client_response(self,
                               base64_images: list[str]) -> ParsedChatCompletion[SeperateCashFlowStatement]:
        """
        Request the OpenAI API to parse image into the SeperateCashFlowStatement format.
        """
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
        """
        A function that create a request to OpenAI API to convert the image into **markdown** or **html** format.
        
        Args:
            base64_images (list[str]): The list of base64-encoded images.
            mode (Literal["html", "markdown"]): The desired output format.
        
        Returns:
            Response: The response object from the OpenAI API
        """
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
                    Do not include the special markdown '```' in the output.
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
                                response: Union[Response, ParsedChatCompletion]) -> BatchResponse:
        """
        A handle function that calculates the cost of the API call based on the response type and model used.

        Args:
            idx (int): The index of the batch.
            response (Union[Response, ParsedChatCompletion]): The response object from OpenAI API.

        Returns:
            BatchResponse: The parsed response object containing the batch index, content received from OpenAI, and cost.
        """


        if isinstance(response, Response):
            # Track the cost of the API call
            cost = CostTracker.track_cost(
                input_tokens_count=response.usage.input_tokens,
                output_tokens_count=response.usage.output_tokens,
                model_name="gpt-4o",
            )

            batch_response = BatchResponse(
                page=idx,
                content=response,
                cost=cost
            )

            return batch_response
        
        elif isinstance(response, ParsedChatCompletion):
            cost = CostTracker.track_cost(
                input_tokens_count=response.usage.prompt_tokens,
                output_tokens_count=response.usage.completion_tokens,
                model_name="gpt-4o"
            )

            batch_response = BatchResponse(
                page=idx,
                content=response,
                cost=cost
            )

            return batch_response
        
    async def _async_image_parser(self, 
                                 idx: int,
                                 images: list[Image.Image],
                                 mode: Literal["html", "markdown", "tabular"] = "tabular"
                                 ) -> BatchResponse:
        """
        The core function that handles the image preprocessing and create request to the OpenAI API for parsing.
        
        Args:
            idx (int): The index of the image.
            images (list[Image.Image]): The list of images.
            mode (Literal["html", "markdown", "tabular"]): The desired output format.
        
        Returns:
            BatchResponse: The parsed response object containing the batch index, content received from OpenAI, and cost.
        """
        
        processed_images = []

        # Image preprocessing logic (rotate and enhance)
        for image in images:
            image = ImageProcessor.rotate_image(
                image=image,
                method="hough"
            )

            # Enhance the image
            image = ImageProcessor.enhance_image(image=image)
            
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

        # Calculate the cost of the API call
        batch_response = self._handle_client_response(
            idx=idx,
            response=response,
        )

        return batch_response
    
    def _handle_responses(self,
                          responses: list[BatchResponse],
                          mode: Literal["html", "markdown", "tabular"] = "tabular") -> ParserResponse:
        """
        Handle the responses usage information based on the response type in the PageReponse object.
        Supported formats: HTML, Markdown, and Tabular.

        Args:
            responses (list[BatchResponse]): The list of BatchResponse objects.
            mode (Literal["html", "markdown", "tabular"]): The desired output format.

        Returns:
            ParserResponse: The parsed response object.
        """
                    # Calculate the total cost of the API calls
        total_cost = sum(response.cost for response in responses)
        
        # Calculate the total number of pages
        total_pages = len(responses)
        
        # Calculate the total number of input tokens based on the response type
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
        """
        Parse the PDF file into the desired format.
        Supported formats: HTML, Markdown, and Tabular.

        Args:
            pdf_path (str): The path to the PDF file.
            batch_size (int): The batch size for the image loader. (default: 3)
            mode (Literal["html", "markdown", "tabular"]): The desired output format. (default: "tabular")
                - "html": Convert the image into HTML format.
                - "markdown": Convert the image into markdown format.
                - "tabular": Convert the image into a pydantic schema format. **(Recommended)**
        
        Returns:
            ParserResponse: The parsed response object.

        ```python
        import asyncio
        from openai import OpenAI
        from dotenv import load_dotenv
        from finast.core import OpenAIPDFParser
        import json

        load_dotenv()

        client = OpenAI()
            
        async def main():
            pdf_parser = OpenAIPDFParser(
                client=client,
                mode="tabular"
            )

            output = await pdf_parser.async_parse_pdf(
                pdf_path="docs/finance_report.pdf"
            )
            
            with open("output.json", "w") as f:
                json.dump(output.model_dump(), f, indent=4, ensure_ascii=False)

        if __name__ == "__main__":
            asyncio.run(main())
        ```
        """

        # Set the mode if provided
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
        responses = sorted(responses, key=lambda x: x.index)

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
    