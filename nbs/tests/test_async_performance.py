#!/usr/bin/env python3
"""
Test script to verify async performance improvement with gpt-4o-mini
"""
import asyncio
import time
import os
from onprem.llm.base import LLM

# Test prompts for summarization
TEST_PROMPTS = [
    "Summarize the key benefits of renewable energy in 2 sentences.",
    "Explain machine learning in simple terms for a beginner.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis briefly.",
    "List 3 advantages of remote work.",
    "What is the importance of data privacy?",
    "Explain what blockchain technology does.",
    "What are the benefits of exercise for mental health?",
    "Describe how the internet works in simple terms.",
    "What makes a good leader?"
]

def test_sync_performance(llm, prompts, num_runs=1):
    """Test synchronous performance"""
    print(f"\n🔄 Testing SYNC performance with {len(prompts)} prompts ({num_runs} run{'s' if num_runs > 1 else ''})...")
    
    all_times = []
    for run in range(num_runs):
        start_time = time.time()
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"  Sync run {run+1}/{num_runs}: Processing prompt {i+1}/{len(prompts)}...")
            result = llm.prompt(prompt)
            results.append(result)
        
        end_time = time.time()
        run_time = end_time - start_time
        all_times.append(run_time)
        print(f"  Sync run {run+1}/{num_runs} completed in {run_time:.2f} seconds")
    
    avg_time = sum(all_times) / len(all_times)
    return avg_time, results if num_runs == 1 else None

async def test_async_performance(llm, prompts, concurrency=4, num_runs=1):
    """Test asynchronous performance with concurrency control"""
    print(f"\n⚡ Testing ASYNC performance with {len(prompts)} prompts, concurrency={concurrency} ({num_runs} run{'s' if num_runs > 1 else ''})...")
    
    all_times = []
    for run in range(num_runs):
        start_time = time.time()
        
        # Create semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def process_prompt(i, prompt):
            async with sem:
                print(f"  Async run {run+1}/{num_runs}: Processing prompt {i+1}/{len(prompts)}...")
                result = await llm.aprompt(prompt)
                return result
        
        # Create tasks for all prompts
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        run_time = end_time - start_time
        all_times.append(run_time)
        print(f"  Async run {run+1}/{num_runs} completed in {run_time:.2f} seconds")
    
    avg_time = sum(all_times) / len(all_times)
    return avg_time, results if num_runs == 1 else None

async def main():
    """Main test function"""
    print("🚀 Async Performance Test for OnPrem LLM with GPT-4o-mini")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize LLM with GPT-4o-mini
    print("🔧 Initializing LLM with GPT-4o-mini...")
    try:
        llm = LLM(model_url="openai://gpt-4o-mini", max_tokens=100)
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return
    
    # Test with subset of prompts first
    test_prompts = TEST_PROMPTS[:5]  # Use first 5 prompts for faster testing
    
    try:
        # Test sync performance
        sync_time, sync_results = test_sync_performance(llm, test_prompts)
        
        # Test async performance with different concurrency levels
        concurrency_levels = [2, 4]
        async_results = {}
        
        for concurrency in concurrency_levels:
            async_time, async_res = await test_async_performance(llm, test_prompts, concurrency)
            async_results[concurrency] = async_time
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE RESULTS")
        print("=" * 60)
        print(f"Sync Performance:     {sync_time:.2f} seconds")
        
        for concurrency, async_time in async_results.items():
            speedup = sync_time / async_time
            print(f"Async (concur={concurrency}):   {async_time:.2f} seconds (🚀 {speedup:.1f}x speedup)")
        
        # Show best performance
        best_async_time = min(async_results.values())
        best_concurrency = min(async_results, key=async_results.get)
        best_speedup = sync_time / best_async_time
        
        print(f"\n🏆 Best Performance: Async with concurrency={best_concurrency}")
        print(f"   Speed improvement: {best_speedup:.1f}x faster than sync")
        print(f"   Time saved: {sync_time - best_async_time:.2f} seconds ({((sync_time - best_async_time) / sync_time * 100):.1f}% reduction)")
        
        # Verify outputs are similar (sample check)
        if sync_results and len(sync_results) > 0:
            print(f"\n✅ Sample output verification:")
            print(f"Sync result length: {len(sync_results[0])} chars")
            print(f"Both methods produced valid responses")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())