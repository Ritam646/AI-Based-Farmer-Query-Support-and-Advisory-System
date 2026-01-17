# from supabase import create_client, Client
# import os

# def create_supabase_client() -> Client:
#     return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

from supabase import create_client, Client
import os

def create_supabase_client() -> Client:
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))