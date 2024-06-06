import ast
import atexit
import datetime
import importlib.metadata
import inspect
import logging
import os
import subprocess
import textwrap
import traceback
from typing import Callable, Union, Dict, Tuple, List, Set, Optional

import fasteners
import pandas as pd
import psutil
from git import Repo
from pydantic import BaseModel, Extra
from rich.logging import RichHandler
from wonderwords import RandomWord

from . import utils

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


class Run(BaseModel, extra=Extra.allow):

    def __init__(
            self,
            params: Optional[dict] = None,
            tags: Optional[Union[List[str], Tuple[str], Set[str]]] = None,
            remote: Optional[Union[str, List[str], Set[str]]] = None,
            verbose: bool = True,
    ):
        """
        Initialize a Run instance.

        Parameters
        ----------
        params : dict, optional
            A dictionary of hyperparameters to track, where the key-value pairs are the hyperparameter names and values.
            Pass `locals()` to automatically capture local variables.
        tags : list, tuple, or set, optional
            Tags to associate with the experiment.
        remote : str, list, or set, optional
            The URL or list of URLs of the remote repositories to save the experiment to.
            Must be valid GitHub, GitLab, or Bitbucket URL for which you have write access.
        verbose : bool, optional
            If True, prints detailed information about the experiment run.

        Example
        -----
        Cubyc offers three ways to define experiment runs: you can explicitly specify the start and end of the run,
        utilize a context manager, or define it as a function.
        All three approaches are equally effective and capture the same information.

        === "Explicit Syntax"

            Create a `Run` and call its `start` and `end` methods to define the start and end of your run.

            ```python linenums="1"
            from cubyc import Run

            model = MLP(hidden_layers=2)
            optimizer = Adam(lr=0.001)

            run = Run(params={"model": model, "opt": optimizer}, tags=["tutorial"])
            run.start()
            for epoch in range(10):
                ...
                run.log({"loss": ..., "acc": ...})

            model.save("models/model.pkl")
            plt.savefig("plot.png")
            run.end()
            ```

        === "Context Syntax"

            Use Python's `with` statement to define a context manager for your experiment.

            ```python linenums="1"
            from cubyc import Run

            model = MLP(hidden_layers=2)
            optimizer = Adam(lr=0.001)

            with Run(params={"model": model, "opt": optimizer}, tags=["tutorial"])):
                for epoch in range(10):
                    ...
                    run.log({"loss": ..., "acc": ...})

                model.save("models/model.pkl")
                plt.savefig("plot.png")
            ```

        === "Function Syntax"

            Define your experiment as a function and use the `@run.track` decorator to track it.

            ```python linenums="1"
            from cubyc import Run

            model = MLP(hidden_layers=2)
            optimizer = Adam(lr=0.001)

            @Run(tags=["tutorial"]))
            def experiment_func(model, optimizer):
                for epoch in range(10):
                    ...
                    yield {"loss": ..., "acc": ...}

                model.save("model.pkl")
                plt.savefig("plot.png")
            experiment_func(model=model, optimizer=optimizer)
            ```
        """

        super().__init__()

        path = utils.get_caller_path()
        if not os.path.exists(os.path.join(path, ".git")) or not os.path.exists(os.path.join(path, ".cubyc")):
            raise ValueError("Please run `cubyc init` to initialize a project in this directory.")

        self.repo = Repo(path=utils.get_caller_path())

        # Parameters
        self._params = utils.serialize(params)

        # Tags
        if tags is not None:
            if not isinstance(tags, (list, tuple, set)) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("Tags must be a list, tuple, or set of strings")

            self._tags = tags
        else:
            self._tags = []

        self._logs = []

        if not os.path.exists(self._repo_path(".cubyc")) or not os.path.exists(self._repo_path(".git")):
            log.error("Please run [green]cubyc init[/green] to initialize a project in this directory.",
                      extra={"markup": True})
            exit()

        self.remotes = set() if remote is None else {remote} if isinstance(remote, str) else remote
        self.remotes |= {remote.url for remote in self.repo.remotes}
        self._verbose = verbose
        self._lock = fasteners.InterProcessLock(self._repo_path(".cubyc/lock.file"))
        self._function = None
        self._function = None
        self._lineno_start = None
        self._lineno_end = None

    def __enter__(self):
        if self._function is not None:
            self._source_code = inspect.getsource(self._function)
            self._lineno_start = 1
        else:
            first_stack = traceback.extract_stack()[0]
            self._lineno_start = first_stack.lineno

            with open(os.path.abspath(os.path.abspath(first_stack.filename)), "r") as file:
                self._source_code = "".join(file.readlines())

        self._timestamp = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._function is not None:
            # Count the number of lines in the source code
            self._lineno_start = 2
            self._lineno_end = -1
        elif self._lineno_end is None:
            self._lineno_end = traceback.extract_stack()[0].lineno

        # self._csv_file.flush()
        # self._csv_file.close()

        with self._lock:
            code = textwrap.dedent("\n".join(self._source_code.split("\n")[self._lineno_start:self._lineno_end]))
            metadata = self._get_metadata(code)
            branch = self.repo.active_branch.name
            code_ast = utils.ast_to_json(utils.IgnorePrintTransformer().visit(ast.parse(code)))

            if branch == "main" or (os.path.exists(self._repo_path(".cubyc/ast.json")) and len(
                    utils.get_changes(utils.load_json(self._repo_path(".cubyc/ast.json")), code_ast))):

                # creates a new branch from main
                branch = Run._get_random_branch_name()
                subprocess.check_output(["git", "checkout", "-b", branch, "-q"], cwd=self.repo.working_dir)
                commit_message = "Code change"
            elif os.path.exists(self._repo_path(".cubyc/config.yaml")):
                changes = utils.get_changes(utils.load_yaml(self._repo_path(".cubyc/config.yaml")),
                                            self._params)
                if len(changes) == 0:
                    commit_message = "No hyperparameter changes"
                elif len(changes) == 1:
                    commit_message = "{} hyperparameter change\n{}".format(len(changes), "\n".join(changes))
                else:
                    commit_message = "{} hyperparameter changes\n{}".format(len(changes), "\n".join(changes))
            else:
                commit_message = "New code"

            logs = pd.DataFrame(self._logs, columns=["timestamp", "name", "value"])

            utils.save_json(data=code_ast, filename=self._repo_path(".cubyc/ast.json"))
            utils.save_yaml(data=self._params, filename=self._repo_path(".cubyc/config.yaml"))
            utils.save_json(data=metadata, filename=self._repo_path(".cubyc/metadata.json"))
            utils.save_csv(data=logs, filename=self._repo_path(".cubyc/logs.csv"))

            # Add and commits all changes to the local repository
            self.repo.git.add(all=True)
            self.repo.index.commit(commit_message)
            hexsha = self.repo.head.commit.hexsha

            if len(self.remotes):

                for remote in self.remotes:
                    domain, _, _ = utils.get_repo_url_details(remote)
                    url = remote.strip(".git")

                    try:
                        subprocess.run(["git", "push", "--set-upstream", remote, branch, '-q'],
                                       cwd=self.repo.working_dir,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        if domain == "github":
                            url += f"/tree/{hexsha}"
                        elif domain == "gitlab":
                            url += f"/-/tree/{hexsha}"
                        elif domain == "bitbucket":
                            url += f"/commits/{hexsha}"

                        if self._verbose:
                            log.info(f"Run [green]{hexsha}[/green] committed to {url}", extra={"markup": True})

                    except subprocess.CalledProcessError:
                        log.error(f"Cannot push changes to {remote}. "
                                  f"Please make sure the repository exists and you have the right permissions")


            else:
                log.info(f"Run [green]{hexsha}[/green] committed locally", extra={"markup": True})

        atexit.unregister(self.__exit__)
        return False

    def __call__(self, func):
        func_args = inspect.getfullargspec(func).args
        self._function = func

        def wrapper(*args, **kwargs):
            # Adds function arguments to the parameter dictionary
            self._params = {**self._params, **kwargs, **{k: v for k, v in zip(func_args, args)}}

            with self:
                result = func(*args, **kwargs)

            if result is not None and hasattr(result, "__iter__"):
                for variables in result:
                    if isinstance(variables, dict):
                        self.log(variables=variables)

        return wrapper

    def start(self):
        """
        Starts the experiment run.

        See Also
        --------
        - [Run.end](#cubyc.Run.end) : Ends an experiment run.

        Examples
        --------
        Place the `start` and `end` functions around the code you want to track.

        ```python
        from cubyc import Run

        run = Run()
        run.start(params={"alpha": 7, "beta", 1e-3}, tags=["example", "start_function"])
        # Your experiment code here
        run.end()
        ```
        """
        return self.__enter__()

    def end(self) -> None:
        """
        Ends an experiment run.

        See Also
        --------
        - [Run.start](#cubyc.Run.start) : Starts an experiment run.
        """
        self._lineno_end = traceback.extract_stack()[0].lineno - 1
        self.__exit__(None, None, None)

    def log(
            self,
            variables: Dict[str, float],
            **kwargs
    ) -> None:
        """
        Logs the specified variable values to the experiment.

        Parameters
        ----------
        variables : dict
            A dictionary of variable, where the key-value pairs are the variable names and values.
        kwargs : dict
            Additional variables to log.

        Examples
        --------
        Call the run's log method with a dictionary containing the metrics you want to track.

        ```python
        from cubyc import run

        run = run(remote="https://github.com/owner/project.git")

        run.start(tags=["example", "log_method"])

        run.log({"accuracy": 0.9, "loss": 0.1})
        ```

        Alternatively, yield a dictionary containing the desired metrics if you are tracking functions.


        ```python
        run = run(remote="https://github.com/owner/project.git")

        @run.track(tags=["example", "log_decorator"])
        def my_function():
            yield {"accuracy": 0.9, "loss": 0.1}
        ```
        """
        timestamp = datetime.datetime.now()

        variables = variables or {}

        for k, v in {**variables, **kwargs}.items():
            self._logs.append((timestamp, k, v))
            # self._csv_writer.writerow([timestamp, k, v])
        # self._csv_file.flush()

    @staticmethod
    def _trace_file_lineno(file: str) -> Tuple[Optional[str], Optional[int]]:
        # Iterate through the stack in reverse to find the bottom-most frame (closest to the entry point)
        for frame_info in traceback.extract_stack():
            if os.path.basename(frame_info.filename) == os.path.basename(file):
                return frame_info.filename, frame_info.lineno
        return None, None

    @staticmethod
    def _get_random_branch_name() -> str:
        r = RandomWord()
        verb = r.word(include_parts_of_speech=["adjectives"])
        noun = r.word(include_parts_of_speech=["nouns"])
        return f"{verb}-{noun}"

    def _get_metadata(self, code: str) -> dict:

        metadata = {
            "code": code,
            "cpu":
                {
                    "cores": psutil.cpu_count(logical=False),
                    "threads": psutil.cpu_count(logical=True),
                    "usage": psutil.cpu_percent()
                },
            "disk_gb":
                {
                    "total": psutil.disk_usage("/").total / 1e9,
                    "used": psutil.disk_usage("/").used / 1e9
                },
            "memory_gb":
                {
                    "total": psutil.virtual_memory().total / 1e9,
                    "available": psutil.virtual_memory().available / 1e9,
                    "used": psutil.virtual_memory().used / 1e9,
                },
            "python":
                {
                    "version": subprocess.check_output(["python", "--version", "-q"],
                                                       cwd=self.repo.working_dir).decode().strip()
                },
            "requirements": [f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()],
            "runtime": (datetime.datetime.now() - self._timestamp).total_seconds(),
            "tags": self._tags,
            "timestamp": str(self._timestamp),
        }

        return metadata

    def _repo_path(self, relative_path: str) -> str:
        return os.path.join(self.repo.working_dir, relative_path)
