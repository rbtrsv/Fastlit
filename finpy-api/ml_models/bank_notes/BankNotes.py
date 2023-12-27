from pydantic import BaseModel

# Class which describes Bank Notes measurements
class BankNote(BaseModel):
    variance: float 
    skewness: float 
    kurtosis: float 
    entropy: float