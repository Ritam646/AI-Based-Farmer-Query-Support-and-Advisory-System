import asyncio
import httpx
import json

SERVER_URL = "http://127.0.0.1:8000"

async def ask_query(query: str):
    """User gives query, gets answer."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{SERVER_URL}/query", json={"query": query})
        if response.status_code == 200:
            result = response.json()
            print("Your Query:", query)
            print("\nAnswer:\n", result["response"])
            print("\n---\nRaw Data Preview:", json.dumps(result["raw_data"], indent=2)[:300] + "...")
            return result
        else:
            print(f"Error: {response.text}")

async def main():
    print("MCP Server Client: Enter queries (e.g., 'rice prices across India today') or 'quit' to exit.")
    while True:
        user_input = input("\nYour query: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            await ask_query(user_input)

if __name__ == "__main__":
    asyncio.run(main())