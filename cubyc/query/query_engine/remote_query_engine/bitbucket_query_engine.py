import asyncio
import io
import itertools
import json
from typing import List, Tuple, Optional

import aiohttp
import pandas as pd
import yaml

from .remote_query_engine import RemoteQueryEngine

BITBUCKET_REST_URL = "https://api.bitbucket.org/2.0/repositories"
BITBUCKET_GRAPHQL_URL = "https://api.atlassian.com/graphql"  # unused


class BitbucketQueryEngine(RemoteQueryEngine):
    async def get_commits(
            self,
            branch: Optional[str] = None
    ) -> List[str]:
        """
        Implementation of get_commits method for the BitbucketQueryEngine class.
        """
        url = f"{BITBUCKET_REST_URL}/{self.owner}/{self.repo_name}/commits?pagelen=100"

        if branch is not None:
            url += f"&include={branch}"

        all_commits = []
        while url:
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                all_commits.extend([commit["hash"] for commit in data["values"]])
                url = data.get("next")

        return all_commits

    async def get_tables(
            self,
            commits: List[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Implementation of get_tables method for the BitbucketQueryEngine class.
        """

        config, metadata, logs, comments = None, None, None, None

        if not self.exclude_config:
            preprocess = lambda sha, data: {**{'id': sha}, **yaml.safe_load(data)}
            config = await asyncio.gather(
                *[self._fetch("config.yaml", c, preprocess, self.owner, self.repo_name) for c in commits])
            config = filter(lambda item: item is not None, config)
            config = pd.DataFrame.from_records(config, coerce_float=True)

        if not self.exclude_metadata:
            preprocess = lambda sha, data: {**{'id': sha}, **json.loads(data)}
            metadata = await asyncio.gather(
                *[self._fetch("metadata.json", c, preprocess, self.owner, self.repo_name) for c in commits])
            metadata = filter(lambda item: item is not None, metadata)
            metadata = pd.DataFrame.from_records(metadata, coerce_float=True)

        if not self.exclude_logs:
            preprocess = lambda sha, data: pd.read_csv(io.StringIO(data)).assign(id=sha)
            logs = await asyncio.gather(
                *[self._fetch("logs.csv", c, preprocess, self.owner, self.repo_name) for c in commits])
            logs = filter(lambda item: item is not None, logs)
            logs = pd.concat(logs, ignore_index=True)[["id", "timestamp", "name", "value"]]

        if not self.exclude_comments:
            comments = await asyncio.gather(*[self._get_comments(self.owner, self.repo_name, c) for c in commits])
            comments = list(itertools.chain(*list(filter(lambda item: item is not None, comments))))
            comments = pd.DataFrame.from_records(comments, columns=["id", "author", "content", "timestamp"])
            if len(comments):
                comments = comments.pivot_table(index=["id"], aggfunc='first').reset_index()
                comments['timestamp'] = pd.to_datetime(comments['timestamp'])
            comments = comments[["id", "timestamp", "author", "content"]]

        return config, metadata, logs, comments

    async def _fetch(self, file: str, sha: str, preprocess: callable, owner: str, repo: str) -> Optional[dict]:
        url = f"{BITBUCKET_REST_URL}/{owner}/{repo}/src/{sha}/.cubyc/{file}"

        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.text()
                if 'error' in data or data is None:
                    return None
                return preprocess(sha, data)
        except aiohttp.ClientResponseError:
            return None

    async def _get_comments(self, owner: str, repo: str, sha: str) -> Optional[List[dict]]:
        url = f"{BITBUCKET_REST_URL}/{owner}/{repo}/commit/{sha}/comments"

        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

                if len(data["values"]):
                    return [{"id": sha,
                             "author": comment["user"]["display_name"],
                             "content": comment["content"]["raw"],
                             "timestamp": comment["created_on"]}
                            for comment in data["values"]]
        except aiohttp.ClientResponseError:
            return None
