from pydantic import BaseModel

class ImageObject(BaseModel):
    base64_image: str

    @property
    def input_image_dict(self):
        return {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{self.base64_image}"
        }
    
    @property
    def image_url_dict(self):
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{self.base64_image}"
            }
        }