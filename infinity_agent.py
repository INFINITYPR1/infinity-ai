import os
import sys
import subprocess
import json
from openai import OpenAI

# 1. Initialize OpenAI client mapped to local Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

MODEL = "llama3.2:latest"

# 2. Define JSON Schema for Structured Outputs
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_action",
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Reasoning behind the current step or diagnosis."
                },
                "code": {
                    "type": ["string", "null"],
                    "description": "Python code script to execute. Set to null if the goal is fully achieved."
                }
            },
            "required": ["thought", "code"],
            "additionalProperties": False
        }
    }
}

# 3. Local Execution Engine
def execute_python_code(code_string: str) -> dict:
    """Saves code to a temporary workspace file and executes it via subprocess."""
    temp_filename = "_infinity_workspace.py"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(code_string)

    try:
        result = subprocess.run(
            [sys.executable, temp_filename],
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Error: Script execution timed out (30s limit exceeded).",
            "exit_code": -1
        }
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# 4. System Prompt
SYSTEM_PROMPT = """You are Infinity, an autonomous self-healing AI agent running locally on macOS.
You have the ability to write and execute Python code to achieve user goals.

You MUST always output valid JSON matching the specified schema containing:
- 'thought': string detailing your step-by-step reasoning or error analysis.
- 'code': string of complete executable Python code, OR null if the objective is fully completed.
"""

# 5. Agent Loop Implementation
def run_infinity_agent(goal: str, max_iterations: int = 5):
    print(f"\n🚀 [Infinity Autonomous Agent Initialized]")
    print(f"🎯 Objective: {goal}\n" + "="*60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Achieve this objective autonomously: {goal}"}
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 [Iteration {iteration}/{max_iterations}] Querying Infinity core...", flush=True)

        try:
            # Enforce schema via response_format parameter
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                response_format=JSON_SCHEMA
            )
            
            raw_content = response.choices[0].message.content.strip()
            parsed = json.loads(raw_content)
            
            thought = parsed.get("thought", "")
            code = parsed.get("code")

            print(f"\n🧠 Infinity Thought:\n{thought}")

            if not code:
                print("\n✅ Infinity reports the task is complete!")
                break

            print("\n💻 Executing Generated Script...")
            exec_result = execute_python_code(code)

            if exec_result["success"]:
                print("🟢 Execution Successful!")
                print(f"STDOUT:\n{exec_result['stdout']}")
                
                # Check if output fulfills goal and exit early if completed
                if exec_result["stdout"].strip():
                    print("\n✅ Task completed successfully with output!")
                    break

                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user",
                    "content": f"Execution Succeeded.\nSTDOUT:\n{exec_result['stdout']}\nIf the objective is satisfied, set 'code' to null in your next JSON response."
                })
            else:
                print("🔴 Execution Error Encountered!")
                print(f"STDERR:\n{exec_result['stderr']}")

                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user",
                    "content": f"Execution Failed with Exit Code {exec_result['exit_code']}.\nSTDERR:\n{exec_result['stderr']}\nAnalyze this error, fix your script, and output the updated JSON."
                })

        except json.JSONDecodeError:
            print("⚠️ Warning: JSON parse failed. Re-prompting schema enforcement...")
            messages.append({
                "role": "user",
                "content": "Return ONLY a raw JSON object with 'thought' and 'code' keys matching the required schema."
            })
        except Exception as e:
            print(f"❌ Error in Agent Loop: {e}")
            break

if __name__ == "__main__":
    test_goal = "Write a Python script that calculates disk space usage, creates a summary JSON file named 'disk_info.json', and reads it back to stdout."
    run_infinity_agent(test_goal)