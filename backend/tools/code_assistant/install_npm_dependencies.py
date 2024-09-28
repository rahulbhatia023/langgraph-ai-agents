import subprocess

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class NpmDependencySchema(BaseModel):
    package_names: str = Field(
        description="Name of the npm packages to install. Should be space-separated."
    )


@tool("install_npm_dependencies", args_schema=NpmDependencySchema, return_direct=True)
def install_npm_dependencies(package_names: str):
    """Installs the given npm dependencies and returns the result of the installation."""
    try:
        package_list = package_names.split()
        command = ["npm", "install"] + package_list
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"Failed to install npm packages '{package_names}': {e.stderr}"

    return f"Successfully installed npm packages '{package_names}'"
