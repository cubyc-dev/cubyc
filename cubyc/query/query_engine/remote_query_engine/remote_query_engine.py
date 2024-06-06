import asyncio
from typing import Tuple, Optional

from ..query_engine import QueryEngine
from .... import utils


class RemoteQueryEngine(QueryEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.path.startswith("http"):
            raise ValueError(f"{self.path} is not a valid URL.")

        self.domain, self.owner, self.repo_name = utils.get_repo_url_details(self.path)

    @staticmethod
    async def manage_file_generators(
            generators: list
    ) -> Tuple[Optional[list], Optional[list], Optional[list], Optional[list]]:
        """
        Manages multiple async generators and collects their results

        Parameters
        ----------
        generators : list
            List of async generators, each yielding a tuple of the form (str, Any)

        Returns
        -------
        Tuple[Optional[list], Optional[list], Optional[list], Optional[list]]:
            A tuple of lists, each containing the results of the generators. The order of the lists is as follows:
            - config
            - metadata
            - logs
            - comments
        """
        tasks = {asyncio.create_task(gen.__anext__()): gen for gen in generators}  # Initial tasks from generators

        logs = []
        metadata = []
        config = []
        comments = []

        try:
            while tasks:
                # Wait for the first task to complete
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    gen = tasks.pop(task)

                    try:
                        result = task.result()  # Get the result

                        if result[0] == "config":
                            config.append(result[1])
                        elif result[0] == "metadata":
                            metadata.append(result[1])
                        elif result[0] == "logs":
                            logs.append(result[1])
                        elif result[0] == "comments":
                            comments.append(result[1])

                        # Schedule the next result from the same generator
                        tasks[asyncio.create_task(gen.__anext__())] = gen
                    except StopAsyncIteration:
                        # Generator is exhausted
                        continue

        finally:
            # Cancel any remaining tasks if they exist
            for task in tasks:
                task.cancel()

        return config, metadata, logs, comments
