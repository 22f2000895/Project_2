import os
import sys
import json
import shutil
import tempfile
import asyncio
import subprocess
import aiohttp
from fastapi import FastAPI, Request, UploadFile, File, Form
from typing import List, Optional
import numpy as np # Import numpy for type checking
import pandas as pd # Import pandas for type checking

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openai/v1"

app = FastAPI()

async def call_llm(questions: str, data_files: dict) -> str:
    """
    Calls an LLM via API to generate a Python script based on user questions and files.
    """
    if not AIPIPE_TOKEN:
        raise ValueError("AIPIPE_TOKEN environment variable not set.")

    # A list of available files to provide context to the LLM
    file_names = list(data_files.keys())

    # The prompt instructs the LLM to act as a data analyst and return only Python code.
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a highly skilled, data analyst agent. Your task is to write a single, complete, executable Python script to answer the user's questions based on the provided context and files. "
                "The script will be executed in a sandboxed environment that has access to the following files: "
                f"{file_names}. "
                "The script must start with the line `# -*- coding: utf-8 -*-` and use valid UTF-8 characters for all strings (especially correct UTF-8 representation for ° is the two-byte sequence \xc2\xb0)."
                "The script must only use the following libraries: **pandas**, **numpy**, **matplotlib**, **requests**, **duckdb**, **BeautifulSoup4**, and **Pillow**."
                "Your process will follow two primary steps: **1. Plan Generation** and **2. Code Generation**. You must first generate a detailed plan and then write the code to execute that plan."
                
                "1. **Plan Generation**: Before writing any code, outline a clear, step-by-step plan in a comment block. This plan should break down the user's request into specific, actionable steps based on the question types (e.g., DuckDB query, web scraping, image processing, data analysis, script execution). For example:"
                "   # Plan:"
                "   # 1. Check for required files and data."
                "   # 2. If a web scraping task is requested, use requests and BeautifulSoup to fetch and parse data from the specified URL. After scraping the data and headers, validate that the number of headers matches the number of columns in each data row. If the scraped headers don't align with the data, create the DataFrame without providing explicit column names, and then use the first row of data as the column names if appropriate. Handle errors gracefully by returning a JSON object with an error key."
                "   # 3. If a DuckDB query is needed, connect to the attached data files and run the query."
                "   # 4. If image processing is required, use Pillow to analyze the image file."
                "   # 5. Perform any necessary data cleaning, filtering, or calculations."
                "   # 6. If a chart is requested, generate it using matplotlib and encode it as a base64 string."
                "   # 7. Package the final results into a JSON array, as per instructions in `questions.txt`."
                "   # 8. Validation: After scraping headers and data, verify that the number of headers matches the number of columns in each data row."
                "   If there is a mismatch, use an alternative method to create the DataFrame or return an error such as having pandas infer the headers, if a direct match fails."
                "   If len(headers) does not equal len(rows[0]), print a clear error message and return a JSON object with a detailed error."
                "   If the number of columns and headers don't match, create the DataFrame by letting pandas infer the column names from the data and log a warning"

                "2. **Code Generation**: Following the plan, write the full, executable Python script. The script must perform the following actions in this order:"
                " - **Load and Inspect Data**: Create a function to load data from attached files, which may include CSV, Excel, PDF, JSON, or image files. Normalize all column names for tabular data using `df.columns = df.columns.str.strip().str.lower()`. Include a try-except block to handle missing required columns."
                " - **Execute Tasks**: Use the approved libraries to perform the analysis specified in the plan. This includes using `duckdb` for SQL, `requests` and `BeautifulSoup4` for web scraping, `Pillow` for images, and `subprocess` to run attached Python scripts if necessary."
                " - **Generate Output**: For all questions, compute the answers and format them as a single Python object. For charts or images, save the figure to an in-memory buffer and encode it as a base64 PNG string (`data:image/png;base64,...`). "
                " - **Format Final Response**: Create a single Python object, which must be a **JSON array**, containing an answer for each question. Strictly adhere to the output format specified in `questions.txt`."
                " - **CRITICAL STEP**: The script **must include** a custom JSON encoder to handle NumPy data types. Use the provided function below: "
                "   ```python"
                "   class NpEncoder(json.JSONEncoder):"
                "       def default(self, obj):"
                "           if isinstance(obj, np.integer):"
                "               return int(obj)"
                "           elif isinstance(obj, np.floating):"
                "               return float(obj)"
                "           elif isinstance(obj, np.ndarray):"
                "               return obj.tolist()"
                "           return super(NpEncoder, self).default(obj)"
                "   ```"
                " - **CRITICAL STEP**: When printing the final output, use this encoder with `json.dumps()`: "
                "   `print(json.dumps(your_results_array, cls=NpEncoder))`"
                " - **CRITICAL STEP**: The script must use valid UTF-8 characters for all strings. For the degree symbol (°), ensure you use the correct UTF-8 character directly or its Unicode escape sequence \u00b0.For example plt.ylabel('Temperature (°C)') as plt.ylabel('Temperature (\u00b0C)') and ensure the script is saved and interpreted as UTF-8"
                "Before performing any calculations, verify that the required columns exist in the DataFrame. If a key column like gross is not found, attempt to find a similar column name (e.g., 'worldwide gross') or return a JSON error object indicating the missing data."
                "After loading any tabular data into a pandas DataFrame, immediately normalize the column names using df.columns = df.columns.str.lower().str.strip() to ensure consistency."
                "Return ONLY the executable Python code, starting with the plan comment block. Do not include any markdown code blocks (```python) or conversational text. If an error occurs, the script should output a default JSON array with an error object in the format specified in `questions.txt`."
            ),
        },
        {
            "role": "user",
            "content": questions,
        },
    ]

    payload = {
        "model": "gpt-4o-mini",
        "messages": prompt_messages,
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{AIPIPE_URL}/chat/completions", headers=headers, json=payload, timeout=120) as response:
                response.raise_for_status()
                llm_response = await response.json()

                # Extract the code from the LLM's response
                script_code = llm_response['choices'][0]['message']['content'].strip()

                return script_code

    except aiohttp.ClientError as e:
        print(f"API request failed: {e}")
        return json.dumps({"error": "LLM API request failed", "details": str(e)})

    except (KeyError, IndexError) as e:
        print(f"Failed to parse LLM response: {e}")
        return json.dumps({"error": "Invalid LLM response format", "details": str(e)})


@app.post("/")
async def analyze_data(request: Request):
    questions_file_name = "questions.txt"
    data_content = {}
    questions = None

    # Await the form() call once to get the form data
    form_data = await request.form()
    
    # Iterate over the values in the Form data object
    for name, form_part in form_data.items():
        if name == questions_file_name:
            questions = await form_part.read()
        elif form_part.filename: # check if the part is a file
            data_content[form_part.filename] = await form_part.read()
    
    if not questions:
        return {"error": "Missing required 'questions.txt' file."}


    llm_script = await call_llm(questions.decode(), data_content)

    temp_dir = tempfile.mkdtemp()

    try:
        script_path = os.path.join(temp_dir, "analysis_script.py")
        with open(script_path, "w") as f:
            f.write(llm_script)

        for filename, content in data_content.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(content)

        # The subprocess execution logic with timeout
        process = subprocess.run(
            [sys.executable, "analysis_script.py"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
            env={"PYTHONIOENCODING": "utf-8", 
                **os.environ,  # Keep existing environment variables
                "MPLCONFIGDIR": temp_dir }
        )

        print(process.stdout)
        result = json.loads(process.stdout)
        return result

    except subprocess.CalledProcessError as e:
        return {"error": "Subprocess execution failed", "details": e.stderr}
    except json.JSONDecodeError as e:
        if "Object of type int64 is not JSON serializable" in process.stdout:
            error_message = "The LLM failed to convert a numeric type (e.g., numpy.int64) to a standard Python type for JSON serialization. This is a common issue with Pandas/Numpy data. The prompt needs to be more explicit."
            return {"error": "Subprocess output was not valid JSON", "details": error_message}
        else:
            return {"error": "Subprocess output was not valid JSON", "details": process.stdout}
    except subprocess.TimeoutExpired as e:
        return {"error": "Subprocess execution timed out", "details": f"Process exceeded the time limit of {e.timeout} seconds."}
    except Exception as e:
        return {"error": "An unexpected error occurred", "details": str(e)}
    finally:
        shutil.rmtree(temp_dir)
