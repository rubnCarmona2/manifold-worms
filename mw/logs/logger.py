from typing import Callable, Any, Type, Union, Literal, Optional
from collections.abc import Mapping, Hashable
from warnings import warn
from types import MethodType


class Logger:
    """
    Implements a framework for category-based processing and routing of logs.
    Any keyword passed to *log* defines a *log category*. Log categories share the same processing pipeline and
    are routed to the same handler method, which must be implemented following the 'handle_<log_category_name>' naming convention.
    Logger requires log categories to hold map[string_name, log_item] objects like *dicts*.

    With **default init settings**, this class:
    - Raises a warning if a logged category doesn't have a correspondent handler. Change 'missing_handler_response' to error to raise an error instead or ignore to ignore it.
    """

    def __init__(
        self,
        missing_handler_response: Literal["error", "warn", "ignore"] = "warn",
        **kwargs,
    ):
        def _attrs_from_dict(**kwargs):
            for key, value in kwargs.items():
                if hasattr(self, key):
                    raise NameError(
                        f"{self.__class__.__name__} can't have an attribute named {key}."
                    )
                if isinstance(value, Callable):
                    value = MethodType(value, self)
                setattr(self, key, value)

        self.__missing_handler_response = missing_handler_response
        self.__handler_cache = {}
        self.log_return()
        _attrs_from_dict(**kwargs)

    def log(self, **categories) -> Any:
        """
        Routes log categories to their correspondent handler methods for processing.
        Expects the class to have a corresponding 'handle_<log_category_name>' method implemented. Make sure 'handle_<log_category_name>' is a valid identifier. Use str's .isidentifier() method if unsure.

        e.g:
        Calling *Logger().log(scalars={'loss': 0.5})* will attempt routing *scalars* to *Logger().handle_scalars*.
        """
        self.preprocess(**categories)
        for name, content in categories.items():
            self.__format_validation(name, content)
            for log_title, value in content.items():
                self.__handler_cache.get(name, self.__find_handler(name))(log_title, value)
        self.postprocess(**categories)
        return self.__return_value

    def log_return(self, value: Optional[Any] = None) -> Any:
        self.__return_value = value

    def __find_handler(self, category: Hashable) -> Callable:
        handler_name = f"handle_{str(category)}"
        handler = getattr(self, handler_name, None)
        if not isinstance(handler, Callable):
            raise TypeError(
                f"Could not find a method called '{handler_name}.'{" '" + handler_name + "' is not an identifier." if not handler_name.isidentifier() else ""}"
            )
        self.register_handler(category, handler)
        return handler

    @staticmethod
    def __format_validation(name: Hashable, logs: Mapping) -> None:
        if not isinstance(logs, Mapping) or not all(
            [isinstance(key, str) for key in logs]
        ):
            raise TypeError(
                "'"
                + name
                + "' doesn't follow the log format convention. Logs must hold map[string_name, log_item] objects like *dicts* (e.g: {'loss' : 0.5})."
            )

    @staticmethod
    def __exception_response(exception: Exception, response_type: str) -> None:
        match response_type:
            case "error":
                raise
            case "warn":
                warn(exception, UserWarning, stacklevel=2)
            case _:
                return

    def clear_handler_cache(self) -> None:
        """
        Mostly useful if doing monkey patching.
        """
        self.__handler_cache.clear()

    def register_handler(
        self, category: Hashable, handler_function: Callable, verbose: bool = False
    ) -> None:
        """
        Adds a handler to cache.
        Done automatically when a handler invoked by a category isn't found in cache.
        """
        if verbose and category in self.__handler_cache:
            warn(
                f"Overriding existing value for {category} in cache.",
                ResourceWarning,
                stacklevel=2,
            )
        handler_name = f"handle_{category}"
        if not isinstance(handler_function, MethodType) or handler_function.__self__ is not self:
            setattr(self, handler_name, MethodType(handler_function, self))
        self.__handler_cache[category] = getattr(self, handler_name)

    def preprocess(self, **categories) -> None:
        """
        Logic that runs once for all log categories before any category is handled or validated.
        Useful for setting state, conditions and formatting for example.
        """
        return

    def postprocess(self, **categories) -> None:
        """
        Logic that runs once for all log categories after all categories have been handled.
        Useful for batch logging for example.
        """
        return
