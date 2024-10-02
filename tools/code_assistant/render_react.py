import os
import queue
import re
import subprocess
import threading
import time

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ReactInputSchema(BaseModel):
    code: str = Field(
        description="Code to render a react component. Should not contain localfile import statements."
    )


@tool("render_react", args_schema=ReactInputSchema, return_direct=True)
def render_react(code: str):
    """Render a react component with the given code and return the render result."""

    file_path = os.path.join(os.getcwd(), "react", "src", "App.js")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        subprocess.run(["pkill", "node"], check=True)
    except subprocess.CalledProcessError:
        pass

    def run_command(command):
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            process.wait()

        except Exception as e:
            return f"An error occurred running command '{' '.join(command)}': {str(e)}"

    run_command(["npm", "--prefix", "./react", "install"])
    run_command(["npm", "--prefix", "./react", "run", "build"])

    output_queue = queue.Queue()
    error_messages = []
    success_pattern = re.compile(r"Compiled successfully|webpack compiled successfully")
    error_pattern = re.compile(r"Failed to compile|Error:|ERROR in")
    start_time = time.time()

    def handle_output(stream, prefix):
        for line in iter(stream.readline, ""):
            output_queue.put(f"{prefix}: {line.strip()}")
        stream.close()

    try:
        process = subprocess.Popen(
            ["npm", "--prefix", "./react", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_thread = threading.Thread(
            target=handle_output, args=(process.stdout, "stdout")
        )
        stderr_thread = threading.Thread(
            target=handle_output, args=(process.stderr, "stderr")
        )

        stdout_thread.start()
        stderr_thread.start()

        compilation_failed = False

        while True:
            try:
                line = output_queue.get(timeout=5)

                if success_pattern.search(line):
                    with open("application.flag", "w") as f:
                        f.write("flag")

                    return f"npm start completed successfully"

                if error_pattern.search(line):
                    compilation_failed = True
                    error_messages.append(line)

                if compilation_failed and "webpack compiled with" in line:
                    return "npm start failed with errors:\n" + "\n".join(error_messages)

            except queue.Empty:
                if time.time() - start_time > 30:
                    return f"npm start process timed out after 30 seconds"

            if not stdout_thread.is_alive() and not stderr_thread.is_alive():
                break

    except Exception as e:
        return f"An error occurred: {str(e)}"

    if error_messages:
        return "npm start failed with errors:\n" + "\n".join(error_messages)

    with open("application.flag", "w") as f:
        f.write("flag")

    return "npm start completed without obvious errors or success messages"
