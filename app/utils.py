from fastapi import Request, HTTPException
from db.engine import get_mongo_collection
from datetime import datetime

def verify_token(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = token.replace("Bearer ", "")
    tokens_collection = get_mongo_collection("api_tokens")
    token_entry = tokens_collection.find_one({"token": token, "active": True})

    if not token_entry:
        raise HTTPException(status_code=403, detail="Invalid or inactive token")

    if datetime.utcnow() > token_entry["expires_at"]:
        raise HTTPException(status_code=403, detail="Token expired")

    return token_entry["owner"]
