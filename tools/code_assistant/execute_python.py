import base64

from e2b_code_interpreter import CodeInterpreter
from langchain_core.tools import tool


@tool
def execute_python(code: str):
    """Execute python code in a Jupyter notebook cell and returns any result, stdout, stderr, display_data, and error."""

    with open("e2b_sandbox.txt", "r") as f:
        sandbox_id = f.read()

    with open("e2b_sandbox.txt", "r") as f:
        e2b_api_key = f.read()

    with CodeInterpreter.reconnect(sandbox_id=sandbox_id, api_key=e2b_api_key) as sandbox:
        execution = sandbox.notebook.exec_cell(code)

        if execution.error:
            print(
                f"There was an error during execution: {execution.error.name}: {execution.error.value}.\n"
            )
            return (
                f"There was an error during execution: {execution.error.name}: {execution.error.value}.\n"
                f"{execution.error.traceback}"
            )

    message = ""
    if execution.results:
        message += "These are results of the execution:\n"
        for i, result in enumerate(execution.results):
            message += f"Result {i + 1}:\n"
            if result.is_main_result:
                message += f"[Main result]: {result.text}\n"
            else:
                message += f"[Display data]: {result.text}\n"
            if result.formats():
                message += f"It has also following formats: {result.formats()}\n"
            if result.png:
                png_data = base64.b64decode(result.png)
                filename = f"chart.png"
                with open(filename, "wb") as f:
                    f.write(png_data)
                print(f"Saved chart to {filename}")

    if execution.logs.stdout or execution.logs.stderr:
        message += "These are the logs of the execution:\n"
        if execution.logs.stdout:
            message += "Stdout: " + "\n".join(execution.logs.stdout) + "\n"
        if execution.logs.stderr:
            message += "Stderr: " + "\n".join(execution.logs.stderr) + "\n"

    return message
