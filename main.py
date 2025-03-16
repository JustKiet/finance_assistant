import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import io
from PIL import Image
from pprint import pprint
from finast.services import OpenAIPDFParser
import json

load_dotenv()

client = OpenAI()
    
async def main():
    pdf_parser = OpenAIPDFParser(
        client=client,
        mode="tabular"
    )

    output = await pdf_parser.async_parse_pdf(
        pdf_path="docs/FLC_Baocaotaichinh_Q3_2022_Congtyme_page_13.pdf"
    )
    
    with open("output/output_page_13.json", "w") as f:
        json.dump(output.model_dump(), f, indent=4, ensure_ascii=False)
    print(json.dumps(output.model_dump(), indent=4))

if __name__ == "__main__":
    asyncio.run(main())