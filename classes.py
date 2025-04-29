from typing import Optional
from pydantic import BaseModel, Field

class ClassifyQuestion(BaseModel):
    request_type: str = Field(description="Classified type of request using 'vendorid' or 'global_question'")
    vendor_id: Optional[str] = Field(None, description="The vendor_id code extracted from the input.")

class FinalResponse(BaseModel):
    answer: str = Field(description="Final processed answer, transformed to uppercase.")

class GlobalResponse(BaseModel):
    answer: str = Field(description="Answer generated for global questions.")

class VendorIDResponse(BaseModel):
    answer: str = Field(description="Fixed response for title questions.")
