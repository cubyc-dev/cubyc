import asyncio
import logging
import os
from typing import Tuple, Optional

import aiohttp
import duckdb
import keyring
import pandas as pd

from .query_engine.local_query_engine import LocalQueryEngine
from .query_engine.remote_query_engine import GitHubQueryEngine, BitbucketQueryEngine, GitLabQueryEngine
from .. import utils

log = logging.getLogger("rich")


def query(
        statement: str,
        path: Optional[str] = None,
        branch: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query a repository of runs using SQL.

    Parameters
    ----------
    statement : str
        A valid SQL statement to query against your repository.
    path : str, optional
        The local path or remote URL of the repository to query. Defaults to the local working directory.
    branch : str, optional
        The branch to query. Defaults to all branches.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the result of the query.

    Raises
    ------
    HTTPError
        If the server returns an error.

    Notes
    -----
    - Use PostgreSQL syntax for the query.
    - You can query the following tables:
        - `config`
        - `metadata`
        - `logs`
        - `comments`
    - All tables have a `id` column that you can use to join them, which represents the commit SHA of the run.

    Example
    --------
    To query the maximum accuracy and the corresponding configuration for a project:

    ```python
    from cubyc import query

    statement = '''
                SELECT
                    config.*,
                    max(logs.value)
                FROM
                    config
                INNER JOIN
                    logs ON config.id = logs.id
                WHERE
                    logs.var = 'accuracy'
                '''

    query(statement=statement, path="https://github.com/owner/project.git")
    ```
    """

    # Check if the tables are included in the query to avoid unnecessary API calls
    lower_statement = statement.lower()
    exclude_config = "config" not in lower_statement
    exclude_metadata = "metadata" not in lower_statement
    exclude_logs = "logs" not in lower_statement
    exclude_comments = "comments" not in lower_statement

    if not exclude_comments and not path.startswith("http"):
        log.error("Comments table can only be queried from remote repositories.")
        exit()

    loop = asyncio.get_event_loop()

    coroutine = _get_tables(path=path or os.getcwd(),
                            exclude_config=exclude_config,
                            exclude_metadata=exclude_metadata,
                            exclude_logs=exclude_logs,
                            exclude_comments=exclude_comments,
                            branch=branch)

    # If the loop is running e.g. an IPython environment, apply nest_asyncio
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()

    config, metadata, logs, comments = loop.run_until_complete(coroutine)

    conn = duckdb.connect()

    try:
        if not exclude_config:
            conn.register("config", config)
        if not exclude_metadata:
            conn.register("metadata", metadata)
        if not exclude_logs:
            conn.register("logs", logs)
        if not exclude_comments:
            conn.register("comments", comments)

        return conn.execute(statement).fetchdf()
    except duckdb.InvalidInputException as e:
        raise e
    finally:
        conn.close()


async def _get_tables(
        path: str,
        exclude_config: bool = False,
        exclude_metadata: bool = False,
        exclude_logs: bool = False,
        exclude_comments: bool = False,
        branch: Optional[str] = None
) -> Tuple:
    """
    Retrieves the tables for a repository by fetching the commits and then the tables.

    Parameters
    ----------
    path : str
        The local path or remote URL of the repository to get tables from.
    exclude_config : bool
        Whether to exclude the config table from the query.
    exclude_metadata : bool
        Whether to exclude the metadata table from the query.
    exclude_logs : bool
        Whether to exclude the logs table from the query.
    exclude_comments : bool
        Whether to exclude the comments table from the query.

    Returns
    -------
    Tuple
        A tuple containing the config, metadata, logs, and comments tables or None if the table is excluded.
    """

    if path.startswith("http"):
        domain, owner, repo_name = utils.get_repo_url_details(path)

        connector = aiohttp.TCPConnector(ssl=False)
        token = keyring.get_password("cubyc", domain)
        headers = {"Authorization": f"Bearer {token}", "User-Agent": "Cubyc", "Accept": "application/json"}

        if branch == "all":
            branch = None
        elif branch in ("main", "master"):
            raise ValueError("Can only query branches other than 'main' or 'master'.")

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:

            if domain == "github":
                engine_class = GitHubQueryEngine
            elif domain == "bitbucket":
                engine_class = BitbucketQueryEngine
            elif domain == "gitlab":
                engine_class = GitLabQueryEngine
            else:
                raise NotImplementedError(f"{domain} is not supported yet.")

            engine = engine_class(path=path, exclude_config=exclude_config, exclude_metadata=exclude_metadata,
                                  exclude_logs=exclude_logs, exclude_comments=exclude_comments, session=session)

            return await engine(branch)
    else:
        engine = LocalQueryEngine(path=path, exclude_config=exclude_config, exclude_metadata=exclude_metadata,
                                  exclude_logs=exclude_logs, exclude_comments=exclude_comments)
        return await engine(branch)
