# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
#     # try:
#     #     payload = {"text": text, "src_lang": src_lang, "tgt_lang": tgt_lang}
#     #     response = requests.post(os.getenv("INDIC_TRANS_URL"), json=payload)
#     #     response.raise_for_status()
#     #     return response.json().get("translation", text)
#     # except Exception as e:
#     #     print(f"[TRANSLATION ERR] {e}")
#          return text  


import requests
import os
from dotenv import load_dotenv

load_dotenv()

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    try:
        payload = {"text": text, "src_lang": src_lang, "tgt_lang": tgt_lang}
        response = requests.post(os.getenv("INDIC_TRANS_URL"), json=payload)
        response.raise_for_status()
        return response.json().get("translation", text)
    except Exception as e:
        print(f"[TRANSLATION ERR] {e}")
        return text  # Fallback to original text