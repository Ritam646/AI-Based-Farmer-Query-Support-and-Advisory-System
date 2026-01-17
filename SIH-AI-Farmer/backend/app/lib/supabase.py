from supabase import create_client, Client
import os

# Load from .env or use hardcoded values
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://agemnwjquzvghevcpzla.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFnZW1ud2pxdXp2Z2hldmNwemxhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTcyNzY1MjEsImV4cCI6MjA3Mjg1MjUyMX0.N6C3t-NQaYCxQfs6OmM0l0Narmy7BbyYHHsIhaFWyMI")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_client():
    return supabase_client