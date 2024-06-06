import asyncio
import io
import json
import math
from typing import List, Tuple, Optional

import ijson
import pandas as pd
import yaml

from cubyc.utils import flatten_dict
from .remote_query_engine import RemoteQueryEngine

GITHUB_REST_URL = "https://api.github.com/repos"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


class GitHubQueryEngine(RemoteQueryEngine):

    def __init__(self, per_page: int = 20, *args, **kwargs):
        self.per_page = per_page
        super().__init__(*args, **kwargs)

    async def get_commits(
            self,
            branch: Optional[str] = None
    ) -> List[str]:
        """
        Implementation of get_commits method for the GitHubQueryEngine class.
        """
        url = f"{GITHUB_REST_URL}/{self.owner}/{self.repo_name}/commits?per_page={self.per_page}"

        if branch is not None:
            url += f"&sha={branch}"

        # Fetch the first page to get pagination details
        async with self.session.get(url) as response:
            response.raise_for_status()

            all_commits = [commit['sha'] for commit in await response.json()]

            link_header = response.headers.get('Link', None)

            if link_header is not None:
                # Extract the total number of pages from the link header
                parts = link_header.split(',')
                last_link = [part for part in parts if 'rel="last"' in part]

                if last_link:
                    last_page_url = last_link[0].split(';')[0].strip(' <>')
                    total_pages = int(last_page_url.split('&page=')[1].split('&')[0])

                    # Generate URLs for all subsequent pages
                    urls = [f"{url}&page={i}" for i in range(2, total_pages + 1)]
                    other_pages_commits = await asyncio.gather(*[self._get_page(page_url) for page_url in urls])
                    for page in other_pages_commits:
                        all_commits.extend(page)

            return all_commits

    async def get_tables(
            self,
            commits: List[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Implementation of get_tables method for the GitHubQueryEngine class.
        """
        batch_size = max(min(math.ceil(len(commits) / 100), 100), 1)  # batch size between 1 and 100

        partitions = [commits[i:i + batch_size] for i in range(0, len(commits), batch_size)]

        config, metadata, logs, comments = await RemoteQueryEngine.manage_file_generators(
            [self._get_files(self.owner, self.repo_name, p) for p in partitions])

        if not self.exclude_config:
            config = pd.DataFrame.from_records(config, coerce_float=True)

        if not self.exclude_metadata:
            metadata = pd.DataFrame.from_records(metadata, coerce_float=True)

        if not self.exclude_logs:
            logs = pd.concat(logs, ignore_index=True)[["id", "timestamp", "name", "value"]]

        if not self.exclude_comments:
            comments = pd.DataFrame.from_records(comments, columns=["id", "author", "content", "timestamp"])
            comments = comments[["id", "timestamp", "author", "content"]]
            if len(comments):
                comments = comments.pivot_table(index=["id"], aggfunc='first').reset_index()
                comments['timestamp'] = pd.to_datetime(comments['timestamp'])

        return config, metadata, logs, comments

    async def _get_page(self, url) -> List[str]:
        async with self.session.get(url) as response:
            data = await response.json()
            return [commit['sha'] for commit in data]

    async def _get_files(self, owner: str, repo: str, shas: List[str]) -> tuple:
        query_parts = []
        for index, sha in enumerate(shas):
            if not self.exclude_config:
                query_parts.append(f"""
                    config_{sha}: object(expression: "{sha}:.cubyc/config.yaml") {{
                        ... on Blob {{
                            text
                        }}
                    }}
                """)
            if not self.exclude_metadata:
                query_parts.append(f"""
                    metadata_{sha}: object(expression: "{sha}:.cubyc/metadata.json") {{
                        ... on Blob {{
                            text
                        }}
                    }}
                """)
            if not self.exclude_logs:
                query_parts.append(f"""
                    logs_{sha}: object(expression: "{sha}:.cubyc/logs.csv") {{
                        ... on Blob {{
                            text
                        }}
                    }}
                """)
            if not self.exclude_comments:
                query_parts.append(f"""
                    comments_{sha}: object(expression: "{sha}") {{
                        ... on Commit {{
                            comments(first: 100) {{
                                nodes {{
                                    body
                                    author {{
                                        login
                                    }}
                                    createdAt
                                }}
                            }}
                        }}
                    }}
                """)

        full_query = f"""
            query {{
                repository(owner: "{owner}", name: "{repo}") {{
                    {''.join(query_parts)}
                }}
            }}
        """

        payload = {"query": full_query}

        async with self.session.post(GITHUB_GRAPHQL_URL, json=payload) as response:
            response.raise_for_status()  # Ensure HTTP errors are raised immediately

            # Use ijson to parse the JSON content directly from the response stream
            async for prefix, event, value in ijson.parse_async(response.content):

                if event == 'string' and prefix.endswith('.text'):
                    # Process each text value based on the file type
                    identifier, sha = prefix.split('.')[2].split("_")

                    if identifier == "config":
                        yield [identifier, {**{"id": sha}, **flatten_dict(yaml.safe_load(value))}]
                    elif identifier == "metadata":
                        yield [identifier, {**{"id": sha}, **flatten_dict(json.loads(value))}]
                    elif identifier == "logs":
                        yield [identifier, pd.read_csv(io.StringIO(value)).assign(id=sha)]
                elif event == 'string' or event == 'number':
                    if prefix.endswith("body"):
                        yield ["comments", {"id": sha, "content": value}]
                    elif prefix.endswith("author.login"):
                        yield ["comments", {"id": sha, "author": value}]
                    elif prefix.endswith("createdAt"):
                        yield ["comments", {"id": sha, "timestamp": value}]
