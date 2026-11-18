import sys
import os

# Import your core agent runner function
from infinity_agent import run_infinity_agent

# Define a suite of test goals with varying difficulty
TEST_CASES = [
    {
        "name": "Missing Import Test",
        "goal": "Write a Python script that calculates the square root of 25 using the math module, prints the result, but intentionally forget to import the math module on the first try so the agent has to self-heal and add 'import math'.",
        "max_iterations": 3
    },
    {
        "name": "Syntax Error Recovery",
        "goal": "Write a Python script that prints 'Hello from Infinity' but includes a deliberate syntax error (like unclosed quotes), then fix it in the next iteration.",
        "max_iterations": 3
    },
    {
        "name": "Data Processing & File Write",
        "goal": "Write a Python script that generates a list of numbers from 1 to 10, calculates their average, and writes the result to a local JSON file named 'avg_result.json'.",
        "max_iterations": 4
    }
]

def run_test_suite():
    print("🚀 Starting Infinity AI Self-Healing Test Suite...\n")
    results = []

    for test in TEST_CASES:
        print(f"----------------------------------------")
        print(f"Running Test: {test['name']}")
        print(f"Goal: {test['goal']}")
        print(f"----------------------------------------")
        
        try:
            # Run the agent loop
            run_infinity_agent(test["goal"], max_iterations=test["max_iterations"])
            results.append({"name": test["name"], "status": "Completed Execution Loop"})
        except Exception as e:
            results.append({"name": test["name"], "status": f"Failed with error: {e}"})

    print("\n=== Test Suite Summary ===")
    for res in results:
        print(f"• {res['name']}: {res['status']}")

if __name__ == "__main__":
    run_test_suite()