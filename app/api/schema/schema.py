from pydantic import BaseModel


class ApiSchema(BaseModel):

    class Config:
        from_attributes = True
