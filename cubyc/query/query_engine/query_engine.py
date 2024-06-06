from abc import ABC
from typing import Optional, Tuple, List

import aiohttp
import pandas as pd


class QueryEngine(ABC):
    """
    Abstract base class for query engines, which are responsible for fetching data from a Git repository.
    """

    def __init__(
            self,
            path: str,
            exclude_config: bool = False,
            exclude_metadata: bool = False,
            exclude_logs: bool = False,
            exclude_comments: bool = False,
            session: aiohttp.ClientSession = None
    ):
        """
        Initializes the query engine.

        Parameters
        ----------
        path : str
            Local path or remote URL to the repository.
        exclude_config : bool
            Whether to exclude the config table from the query.
        exclude_metadata : bool
            Whether to exclude the metadata table from the query.
        exclude_logs :
            Whether to exclude the logs table from the query.
        exclude_comments :
            Whether to exclude the comments table from the query.
        session :
            The aiohttp session to use for making requests.
        """
        self.path = path
        self.session = session
        self.exclude_config = exclude_config
        self.exclude_metadata = exclude_metadata
        self.exclude_logs = exclude_logs
        self.exclude_comments = exclude_comments

    async def __call__(
            self,
            branch: Optional[str] = None,
    ) -> Tuple:
        """
        Retrieves the tables for a repository by fetching the commits and then the tables.

        Parameters
        ----------
        branch : str
            The branch to fetch the tables from. If None, returns the tables for all branches

        Returns
        -------
        Tuple
            A tuple containing the config, metadata, logs, and comments tables or None if the table is excluded.
        """
        return await self.get_tables(await self.get_commits(branch))

    async def get_commits(
            self,
            branch: Optional[str] = None
    ) -> List[str]:
        """
        Fetches the commits list for a repository.

        Parameters
        ----------
        branch : str
            The branch to fetch the commits for. If None, returns the commits for all branches.
        Returns
        -------
        List[str]
            The list of commit hashes.
        """
        raise NotImplementedError

    async def get_tables(
            self,
            commits: List[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetches the tables for a repository.

        Parameters
        ----------
        commits : List[str]
            The list of commit hashes to fetch the tables for.

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            A tuple containing the config, metadata, logs, and comments tables or None if the table is excluded.
        """
        raise NotImplementedError
