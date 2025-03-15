import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import io
from PIL import Image
from pprint import pprint
from finast.components import PDFParser

load_dotenv()

client = OpenAI()

AZURE_DEPLOYMENT_NAME = "gpt-4o"
    
async def main():
    pdf_parser = PDFParser(client)
    output = await pdf_parser.async_parse_pdf("FLC_Baocaotaichinh_Q3_2022_Congtyme_page_11.pdf")
    
    with open("output_page_11.md", "w") as f:
        f.write(output.markdown)
    pprint(output)

if __name__ == "__main__":
    asyncio.run(main())