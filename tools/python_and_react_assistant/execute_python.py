import base64
from typing import Union, Dict

from e2b_code_interpreter import Sandbox
from langchain_core.tools import BaseTool
from pydantic import Field


class ExecutePythonTool(BaseTool):
    e2b_api_key: str = Field(..., description="The E2B API key to use.")

    name: str = "execute-python"
    description: str = (
        "Execute python code in a Jupyter notebook cell and returns any result, stdout, stderr, display_data, and error."
    )

    def _run(self, code: str) -> Union[Dict, str]:
        with Sandbox(api_key=self.e2b_api_key) as sandbox:
            execution = sandbox.run_code(code)

            if execution.error:
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

        if execution.logs.stdout or execution.logs.stderr:
            message += "These are the logs of the execution:\n"
            if execution.logs.stdout:
                message += "Stdout: " + "\n".join(execution.logs.stdout) + "\n"
            if execution.logs.stderr:
                message += "Stderr: " + "\n".join(execution.logs.stderr) + "\n"

        return message
