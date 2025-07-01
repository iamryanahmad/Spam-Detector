from pydantic import BaseModel

class SMS(BaseModel):
    text: str