from abc import ABC
from enum import Enum
from os import path
from typing import Callable, Iterable, Union, Any

from attr import attrs, attrib

from boba_python_utils.general_utils.console_util import hprint_message, eprint_message
from boba_python_utils.io_utils.pickle_io import pickle_save, pickle_load


@attrs(slots=True)
class WorkflowStep:
    run = attrib(type=Callable)
    name = attrib(type=str, default=None)
    enable_optional_post_step = attrib(type=bool, default=True)
    enable_step_result_save = attrib(type=bool, default=False)

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)


class StepResultSaveOptions(str, Enum):
    NoSave = 'no_save'
    Always = 'always'
    OnError = 'on_error'


# ! When using attrs,
# ! always set this to False for a common base class
# ! to enable multi-inheritance,
# ! because Python does not allow mult-inheritance from multiple slot classes.
@attrs(slots=False)
class Workflow(ABC):
    _steps = attrib(type=Iterable[Union[Callable, WorkflowStep]], default=None)
    enable_optional_post_step = attrib(type=bool, default=True)
    enable_step_result_save = attrib(
        type=Union[StepResultSaveOptions, bool, str],
        default=StepResultSaveOptions.OnError
    )
    continue_saved_step_results = attrib(type=bool, default=False)

    def run(self, *args, **kwargs):
        if not self._steps:
            return

        start_step_i = -1
        if self.continue_saved_step_results is not False:

            for i in range(
                    (
                            self.continue_saved_step_results
                            if isinstance(self.continue_saved_step_results, int)
                            else len(self._steps) - 1
                    ), -1, -1
            ):
                step_result_path = self._get_step_output_path(i, *args, **kwargs)
                exists_step_result_or_preloaded_step_result = self._exists_step_result(
                    step_index=i, result_path=step_result_path
                )
                if (
                        exists_step_result_or_preloaded_step_result is not None and
                        exists_step_result_or_preloaded_step_result is not False
                ):
                    hprint_message(f'step {i} result exists', True)
                    start_step_i = i
                    break
                else:
                    hprint_message(f'step {i} result exists', False)

        if start_step_i != -1:
            step_result = self._load_step_result(
                step_index=start_step_i,
                result_path_or_preloaded_result=(
                    step_result_path if
                    isinstance(exists_step_result_or_preloaded_step_result, bool)
                    else exists_step_result_or_preloaded_step_result
                )
            )

        for i in range(start_step_i + 1, len(self._steps)):
            this_step = self._steps[i]
            try:
                if i > 1:
                    prev_step_result = step_result
                step_result = this_step(*args, **kwargs)
            except Exception as err:
                step_save_on_error_enabled = (
                        i > 0 and
                        (not isinstance(self.enable_optional_post_step, bool)) and
                        self.enable_optional_post_step == StepResultSaveOptions.OnError
                )

                eprint_message(
                    'step failed', i,
                    'step_save_on_error_enabled', step_save_on_error_enabled,

                )

                if step_save_on_error_enabled:
                    self._save_step_result(
                        prev_step_result,
                        output_path=self._get_step_output_path(i, *args, **kwargs)
                    )

                raise err

            _step_result = self._post_step(step_result, *args, **kwargs)
            if _step_result is not None:
                step_result = _step_result

            if (
                    self.enable_optional_post_step and
                    (
                            not isinstance(this_step, WorkflowStep) or
                            this_step.enable_optional_post_step
                    )
            ):
                _step_result = self._optional_post_step(step_result, *args, **kwargs)
                if _step_result is not None:
                    step_result = _step_result
            if (
                    (
                            self.enable_step_result_save is True or
                            self.enable_step_result_save == StepResultSaveOptions.Always
                    ) and
                    (
                            not isinstance(this_step, WorkflowStep) or
                            this_step.enable_step_result_save
                    )
            ):
                self._save_step_result(
                    step_result,
                    output_path=self._get_step_output_path(i, *args, **kwargs)
                )

        return step_result

    def _post_step(self, step_result, *args, **kwargs):
        pass

    def _optional_post_step(self, step_result, *args, **kwargs):
        pass

    def _save_step_result(self, step_result, output_path: str):
        pickle_save(step_result, output_path)

    def _load_step_result(self, step_index: int, result_path_or_preloaded_result: Union[str, Any]):
        return pickle_load(result_path_or_preloaded_result)

    def _exists_step_result(self, step_index: int, result_path: str) -> Union[bool, Any]:
        return path.exists(result_path)

    def _get_step_output_path(self, step_index, *args, **kwargs) -> str:
        raise NotImplementedError
