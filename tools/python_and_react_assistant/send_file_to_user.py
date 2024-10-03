import os
from typing import Union, Dict, Annotated, Optional

from e2b_code_interpreter import CodeInterpreter
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation


class SendFilePath(BaseModel):
    filepath: str = Field(description="Path of the file to send to the user.")


class SendFileToUserTool(BaseTool):
    e2b_sandbox_id: str = Field(..., description="The sandbox ID to use.")
    e2b_api_key: str = Field(..., description="The E2B API key to use.")

    name: str = "send-file-to-user"
    description: str = "Send a single file to the user."
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = SendFilePath
    return_direct: bool = True

    def _run(self, filepath: str) -> Union[Dict, str]:
        with CodeInterpreter.reconnect(
            sandbox_id=self.e2b_sandbox_id, api_key=self.e2b_api_key
        ) as sandbox:
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
