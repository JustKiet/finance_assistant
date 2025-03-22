import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import io
from PIL import Image
from pprint import pprint
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