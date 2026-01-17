# from pydantic import BaseModel

# class QueryRequest(BaseModel):
#     query: str

# class MarketQuery(BaseModel):
#     query: str

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class MarketQuery(BaseModel):
    query: str

class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str