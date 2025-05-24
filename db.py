# db.py
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env
load_dotenv()

# Get the database URL from environment
DATABASE_URL = os.getenv("SUPABASE_DB_URL")

if not DATABASE_URL:
    raise ValueError("SUPABASE_DB_URL is not set in the environment variables.")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

# Test connection
if __name__ == "__main__":
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        print("Database connection successful!", cur.fetchone())
        cur.close()
        conn.close()
    except Exception as e:
        print("Database connection failed:", e)