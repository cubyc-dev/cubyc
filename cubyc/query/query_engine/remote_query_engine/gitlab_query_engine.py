import asyncio
import io
import itertools
import json
from typing import List, Tuple, Union, Optional
from urllib import parse

import ijson
import pandas as pd
import yaml

from cubyc.utils import flatten_dict
from .remote_query_engine import RemoteQueryEngine

GITLAB_REST_URL = "https://gitlab.com/api/v4/projects"
GITLAB_GRAPHQL_URL = "https://gitlab.com/api/graphql"


class GitLabQueryEngine(RemoteQueryEngine):

    async def get_commits(
            self,
            branch: Optional[str] = None
    ) -> List[str]:
        """
        Implementation of get_commits method for the GitLabQueryEngine class.
        """
        project_name = parse.quote_plus(f"{self.owner}/{self.repo_name}")
        url = f"{GITLAB_REST_URL}/{project_name}/repository/commits?per_page=100&page=1&"

        if branch is not None:
            url += f"ref_name={branch}"

        all_commits = []
        while url:
            async with self.session.get(url) as response:
                response.raise_for_status()
                all_commits.extend([commit["id"] for commit in await response.json()])

                next_page = response.headers.get("x-next-page")
                if next_page is not None and next_page != "":
                    url = f"{GITLAB_REST_URL}/{project_name}/repository/commits?per_page=100&page={next_page}"
                else:
                    url = None

        return all_commits

    async def get_tables(
            self,
            commits: List[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Implementation of get_tables method for the GitLabQueryEngine class.
        """
        non_excluded = [self.exclude_config, self.exclude_metadata, self.exclude_logs].count(False)

        if non_excluded:
            # batch size needs to be capped to avoid GraphQL query complexity limit
            batch_size = 8 * non_excluded
            partitions = [commits[i:i + batch_size] for i in range(0, len(commits), batch_size)]
            config, metadata, logs, _ = await RemoteQueryEngine.manage_file_generators(
                [self._get_files(self.owner, self.repo_name, p) for p in partitions])

            if not self.exclude_config:
                config = pd.DataFrame.from_records(config, coerce_float=True)
            if not self.exclude_metadata:
                metadata = pd.DataFrame.from_records(metadata, coerce_float=True)
            if not self.exclude_logs:
                logs = pd.concat(logs, ignore_index=True)[["id", "timestamp", "name", "value"]]
        else:
            config, metadata, logs = None, None, None

        if not self.exclude_comments:
            comments = await asyncio.gather(*[self._get_comments(self.owner, self.repo_name, c) for c in commits])
            comments = list(itertools.chain(*list(filter(lambda item: item is not None, comments))))
            comments = pd.DataFrame.from_records(comments, columns=["id", "author", "content", "timestamp"])
            comments = comments.pivot_table(index=["id", "author", "content", "timestamp"],
                                            aggfunc='first').reset_index()
            comments['timestamp'] = pd.to_datetime(comments['timestamp'])
            comments = comments[["id", "timestamp", "author", "content"]]
        else:
            comments = None

        return config, metadata, logs, comments

    async def _get_files(self, owner: str, repo: str, shas: List[str]) -> tuple:
        query_parts = []
        for index, sha in enumerate(shas):
            if not self.exclude_config:
                query_parts.append(f"""
                    config_{sha}: blobs(ref: "{sha}", paths: [".cubyc/config.yaml"]) {{
                      nodes {{
                        ... on RepositoryBlob {{
                          rawTextBlob
                        }}
                      }}
                    }}
                """)
            if not self.exclude_metadata:
                query_parts.append(f"""
                    metadata_{sha}: blobs(ref: "{sha}", paths: [".cubyc/metadata.json"]) {{
                      nodes {{
                        ... on RepositoryBlob {{
                          rawTextBlob
                        }}
                      }}
                    }}
                """)
            if not self.exclude_logs:
                query_parts.append(f"""
                    logs_{sha}: blobs(ref: "{sha}", paths: [".cubyc/logs.csv"]) {{
                      nodes {{
                        ... on RepositoryBlob {{
                          rawTextBlob
                        }}
                      }}
                    }}
                """)

        full_query = f"""
            query {{
                project(fullPath: "{owner}/{repo}") {{
                    repository {{ 
                        {''.join(query_parts)} 
                    }}
                }}
            }}
        """

        payload = {"query": full_query}

        async with self.session.post(GITLAB_GRAPHQL_URL, json=payload) as response:
            response.raise_for_status()  # Ensure HTTP errors are raised immediately

            # Use ijson to parse the JSON content directly from the response stream
            async for prefix, event, value in ijson.parse_async(response.content):

                if event == 'string' and prefix.endswith('.rawTextBlob'):
                    # Process each text value based on the file type
                    identifier, sha = prefix.split('.')[3].split("_")

                    if identifier == "config":
                        yield [identifier, {**{"id": sha}, **flatten_dict(yaml.safe_load(value))}]
                    elif identifier == "metadata":
                        yield [identifier, {**{"id": sha}, **flatten_dict(json.loads(value))}]
                    elif identifier == "logs":
                        yield [identifier, pd.read_csv(io.StringIO(value)).assign(id=sha)]

    async def _get_comments(self, owner: str, repo: str, sha: str) -> Union[List[dict], None]:
        encoded_project_name = parse.quote_plus(f"{owner}/{repo}")
        url = f"{GITLAB_REST_URL}/{encoded_project_name}/repository/commits/{sha}/comments"

        async with self.session.get(url) as response:
            response.raise_for_status()
            data = await response.json()

            if len(data):
                return [{"id": sha,
                         "author": comment["author"]["name"],
                         "content": comment["note"],
                         "timestamp": comment["created_at"]}
                        for comment in data]
