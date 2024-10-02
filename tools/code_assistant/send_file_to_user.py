import os

from e2b_code_interpreter import CodeInterpreter
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SendFilePath(BaseModel):
    filepath: str = Field(description="Path of the file to send to the user.")


@tool("send_file_to_user", args_schema=SendFilePath, return_direct=True)
def send_file_to_user(filepath: str):
    """Send a single file to the user."""

    with open("e2b_sandbox.txt", "r") as f:
        sandbox_id = f.read()

    with open("e2b_api_key.txt", "r") as f:
        e2b_api_key = f.read()

    with CodeInterpreter.reconnect(sandbox_id=sandbox_id, api_key=e2b_api_key) as sandbox:
        remote_file_path = "/home/user/" + filepath
        try:
            file_in_bytes = sandbox.download_file(remote_file_path)
        except Exception as e:
            return f"An error occurred: {str(e)}"
        if not os.path.exists("downloads"):
            os.makedirs("downloads")
        with open(f"downloads/{filepath}", "wb") as f:
            f.write(file_in_bytes)

        return "File sent to the user successfully."
