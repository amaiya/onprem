import asyncio
from onprem.llm.base import LLM

async def test_async_processing():
    # Initialize with cloud model
    llm = LLM(model_url="openai/gpt-4o-mini")
    
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Natural language processing helps computers understand and interpret human language in a valuable way."
    ]
    
    sem = asyncio.Semaphore(4)  # max 4 concurrent requests
    
    async def summarize(doc):
        async with sem:
            print(f"Processing: {doc[:30]}...")
            result = await llm.aprompt(f"Assign a topic keyphrase to this text: {doc}")
            print(f"Result: {result}")
            return result
    
    print("Starting concurrent processing...")
    results = await asyncio.gather(*[summarize(doc) for doc in documents])
    
    print("\nAll results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(test_async_processing())
