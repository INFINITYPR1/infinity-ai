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

MODEL = "llama3.2"

# 2. Local Execution Engine
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

# 3. System Prompt for Self-Healing Tool Use
SYSTEM_PROMPT = """You are Infinity, an autonomous self-healing AI agent running locally on macOS.
You have the ability to write and execute Python code to achieve user goals.

Response Format:
You MUST respond using valid JSON with two fields:
1. "thought": Your reasoning about the current step, errors, or plan.
2. "code": The complete Python script to execute. If the goal is fully met, set "code" to null.

Example JSON output:
{
  "thought": "I need to generate a script that fetches systemic metrics and saves them to a file.",
  "code": "import os\\nprint(os.listdir('.'))"
}

Do not wrap your output in markdown backticks outside of JSON. Output strictly raw JSON.
"""

# 4. Agent Loop Implementation
def run_infinity_agent(goal: str, max_iterations: int = 5):
    print(f"\n🚀 [Infinity Autonomous Agent Initialized]")
    print(f"🎯 Objective: {goal}\n" + "="*60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Achieve this objective autonomously: {goal}"}
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 [Iteration {iteration}/{max_iterations}] Querying Infinity core...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Clean possible markdown wrapping around JSON
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

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
                
                # Append execution feedback to history
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user",
                    "content": f"Execution Succeeded.\nSTDOUT:\n{exec_result['stdout']}\nIf the objective is completely satisfied, set 'code' to null in your next JSON response."
                })
            else:
                print("🔴 Execution Error Encountered!")
                print(f"STDERR:\n{exec_result['stderr']}")

                # Append error log back into conversation for self-healing
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user",
                    "content": f"Execution Failed with Exit Code {exec_result['exit_code']}.\nSTDERR:\n{exec_result['stderr']}\nAnalyze this error, fix your script, and output the updated JSON."
                })

        except json.JSONDecodeError:
            print("⚠️ Warning: Failed to parse Infinity's output as JSON. Retrying prompt structure...")
            messages.append({
                "role": "user",
                "content": "Your response was not valid JSON. Ensure you return ONLY a raw JSON object with 'thought' and 'code' keys."
            })
        except Exception as e:
            print(f"❌ Error in Agent Loop: {e}")
            break

# 5. Run an Objective
if __name__ == "__main__":
    # Test objective requiring script creation, debugging, or system inspection
    test_goal = "Write a Python script that calculates disk space usage, creates a summary JSON file named 'disk_info.json', and reads it back to stdout."
    run_infinity_agent(test_goal)