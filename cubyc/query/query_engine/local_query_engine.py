import asyncio
import io
import json
import os.path
import subprocess
from typing import List, Tuple, Optional

import pandas as pd
import yaml

from .query_engine import QueryEngine


class LocalQueryEngine(QueryEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.path = os.getcwd() if self.path is None or self.path == "." else self.path

        if not os.path.isabs(self.path):
            self.path = os.path.abspath(self.path)

        if not os.path.isdir(self.path) or \
                not os.path.exists(os.path.join(self.path, ".git")) or \
                not os.path.exists(os.path.join(self.path, ".cubyc")):
            raise ValueError(f"{self.path} is not a valid Git directory.")

    async def get_commits(
            self,
            branch: Optional[str] = None,
    ) -> List[str]:
        """
        Implementation of get_commits method for the LocalQueryEngine class.
        """

        command = ["git", "log", "--format=%H"]

        if branch is not None and branch != "all":
            command.append(branch)

        return subprocess.check_output(command, cwd=self.path).decode("utf-8").split("\n")[0:-1]

    async def get_tables(
            self,
            commits: List[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Implementation of get_tables method for the LocalQueryEngine class.
        """

        config, metadata, logs = None, None, None

        if not self.exclude_config:
            preprocess = lambda sha, data: {**{'id': sha}, **yaml.safe_load(data)}
            config = await asyncio.gather(*[self._fetch("config.yaml", c, preprocess) for c in commits])
            config = filter(lambda item: item is not None, config)
            config = pd.DataFrame.from_records(config, coerce_float=True)

        if not self.exclude_metadata:
            preprocess = lambda sha, data: {**{'id': sha}, **json.loads(data)}
            metadata = await asyncio.gather(*[self._fetch("metadata.json", c, preprocess) for c in commits])
            metadata = filter(lambda item: item is not None, metadata)
            metadata = pd.DataFrame.from_records(metadata, coerce_float=True)

        if not self.exclude_logs:
            preprocess = lambda sha, data: pd.read_csv(io.StringIO(data)).assign(id=sha)
            logs = await asyncio.gather(*[self._fetch("logs.csv", c, preprocess) for c in commits])
            logs = filter(lambda item: item is not None, logs)
            logs = pd.concat(logs, ignore_index=True)[["id", "timestamp", "name", "value"]]

        return config, metadata, logs, None

    async def _fetch(self, file: str, sha: str, preprocess: callable) -> Optional[dict]:
        try:
            data = subprocess.check_output(['git', 'show', f'{sha}:.cubyc/{file}'], cwd=self.path).decode()
            return preprocess(sha, data)
        except subprocess.CalledProcessError:
            pass
