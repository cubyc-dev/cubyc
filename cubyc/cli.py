#!/usr/bin/env python3

import os
import subprocess
from enum import Enum
from importlib import resources
from typing import Optional

import keyring
import typer
from rich import print
from tabulate import tabulate
from typing_extensions import Annotated

from cubyc.query import query as cubyc_query

app = typer.Typer()


class PLATFORM(str, Enum):
    github = "github"
    gitlab = "gitlab"
    bitbucket = "bitbucket"


@app.command()
def cleanup() -> None:
    """
    Cleans up the Cubyc repository in the current working directory by removing the .cubyc directory and Git-related files.
    """
    if os.path.exists(".cubyc"):
        subprocess.run(["rm", "-rf", ".cubyc"], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        # Removes Git-related files only if the .cubyc directory exists
        if os.path.exists(".git"):
            subprocess.run(["rm", "-rf", ".git"], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if os.path.exists(".gitignore"):
            subprocess.run(["rm", ".gitignore"], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        print(f"Successfully cleaned up Cubyc repository in {os.getcwd()}")
    else:
        print(f"No Cubyc repository to clean up in {os.getcwd()}")


@app.command()
def init() -> None:
    """
    Initializes a new Cubyc project in the current directory that tracks run to the specified URL.
    """
    if os.path.exists(".cubyc"):
        raise typer.BadParameter("A Cubyc project already exists in this directory.")
    else:
        os.makedirs(".cubyc")

    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True, stdout=subprocess.DEVNULL)

    # creates .gitignore file from template if it doesn't exist
    if not os.path.exists(".gitignore"):
        with resources.open_text("cubyc.templates", "gitignore_template") as file:
            template = file.read()

        with open(".gitignore", "w") as f:
            f.write(template)

    print(f"Initialized empty Cubyc repository in {os.getcwd()}")


@app.command()
def login(
        platform: Annotated[
            PLATFORM, typer.Argument(help="The repository hosting platform to save the access token for.",
                                     case_sensitive=False)],
        token: Annotated[str, typer.Argument(help="The access token to save.")]
) -> None:
    """
    Saves the access token for the specified repository hosting platform in the system keyring.
    """
    if keyring.get_password("cubyc", platform.value) is None:
        keyring.set_password("cubyc", platform.value, token)
    else:
        keyring.delete_password("cubyc", platform.value)
        keyring.set_password("cubyc", platform.value, token)

    if platform == "github":
        color = "#2dba4e"
    elif platform == "gitlab":
        color = "#FC6D26"
    else:
        color = "#0052CC"

    print(f"Successfully saved token for [{color}]{platform}[/{color}]")


@app.command()
def query(
        statement: Annotated[str, typer.Argument(help="SQL query statement to execute")],
        path: Annotated[Optional[str], typer.Option("--path", "-p",
                                                    help="Local path or remote URL of the repository to query")] = '.',
        branch: Annotated[
            Optional[str], typer.Option("--branch", "-b", help="Branch of the repository to query")] = "all",
        pretty: Annotated[bool, typer.Option("--pretty", "-p", help="Prettify the output with tabulation")] = False
) -> None:
    """
    Queries a remote or local repository and prints the results to the console.
    """

    if branch == "all":
        try:
            # Run the git command to get the current branch name
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, check=True)
            # The branch name will be in result.stdout
            branch = result.stdout.strip()

        except subprocess.CalledProcessError as e:
            raise typer.BadParameter(
                "Could not determine the current branch name. Please specify a branch name manually.")

    query = cubyc_query(path=path, statement=statement, branch=branch)

    if pretty:
        print(tabulate(cubyc_query(path=path, statement=statement, branch=branch), headers='keys', tablefmt='psql'))
    else:
        print(query)


if __name__ == "__main__":
    app()
