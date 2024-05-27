from typing import Union, Tuple, Mapping, Sequence, List, Optional, Iterable

from pyspark.sql import Column, DataFrame

PathOrDataFrame = Union[str, DataFrame]
NameOrColumn = Union[str, Column]
NamesOrColumns = Iterable[NameOrColumn]
AliasAndColumn = Union[Tuple[str, NameOrColumn], Mapping[str, NameOrColumn]]
AliasAndColumnName = Union[Tuple[str, str], Mapping[str, str]]
AliasesAndColumns = Union[Sequence[Tuple[str, NameOrColumn]], Mapping[str, NameOrColumn]]
AliasesAndColumnNames = Union[Sequence[Tuple[str, str]], Mapping[str, str]]

ColumnOrAliasedColumn = Union[NameOrColumn, AliasAndColumn]
ColumnNameOrAliasedColumnName = Union[str, AliasAndColumnName]
ColumnsOrAliasedColumns = Union[NamesOrColumns, AliasesAndColumns]
ColumnNamesOrAliasedColumnNames = Union[Iterable[str], AliasesAndColumnNames]


def resolve_name_or_column(arg: Union[NameOrColumn, AliasAndColumn]) -> NameOrColumn:
    """
    Resolve and return the alias if an `AliasAndColumn` is provided, or directly return
    the `NameOrColumn` if such is provided.

    For the input:
    - If it is a `NameOrColumn`, it directly returns the input.
    - If it is an `AliasAndColumn`, it can be one of two types:
        1. A singleton dictionary where the key is the alias. In this case, the function returns the alias.
        2. A tuple where the first element is the alias. Again, the function returns the alias.

    If the input is neither a `NameOrColumn` nor an `AliasAndColumn`, a ValueError is raised.

    Args:
        arg: The input to be resolved, either a `NameOrColumn` (str or Column) or an `AliasAndColumn`
        (a singleton dictionary or a tuple).

    Returns:
        A string that is either the alias from the `AliasAndColumn`, or the `NameOrColumn` itself.

    Raises:
        ValueError: If the input is neither a `NameOrColumn` nor an `AliasAndColumn`.

    Examples:
        column = Column("column_name")

        print(resolve_name_or_column("column_name"))  # Outputs: "column_name"
        print(resolve_name_or_column(column))  # Outputs: column

        print(resolve_name_or_column(("alias", "column_name")))  # Outputs: "alias"
        print(resolve_name_or_column(("alias", column)))  # Outputs: "alias"

        print(resolve_name_or_column({"alias": "column_name"}))  # Outputs: "alias"
        print(resolve_name_or_column({"alias": column}))  # Outputs: "alias"
    """
    if isinstance(arg, Mapping):
        return next(iter(arg))  # returns the first (and only) alias
    if isinstance(arg, Tuple):
        return arg[0]  # returns the alias
    if isinstance(arg, (str, Column)):
        return arg
    raise ValueError("Input must be either a `NameOrColumn` or an `AliasAndColumn`.")


def resolve_names_or_columns(
        arg: Union[Sequence[NameOrColumn], AliasesAndColumns]
) -> Optional[List[NameOrColumn]]:
    """
    Resolve and return a list of aliases if `AliasesAndColumns` is provided or
    directly return the `Sequence[NameOrColumn]`.

    - If the argument is None, it returns None.
    - If it's a Mapping, it returns a list of the keys.
    - If it's a Sequence of Tuple, it returns a list of the first elements from each Tuple.
    - If it's a Sequence of `NameOrColumn`, it directly returns the list of `NameOrColumn`.

    Raises:
        ValueError: If the input is not None, a Mapping, a Sequence of Tuple,
        or a Sequence of NameOrColumn.

    Examples:
        print(resolve_names_or_columns(None))  # Outputs: None

        print(resolve_names_or_columns({"alias1": "column_name1", "alias2": "column_name2"}))
        # Outputs: ['alias1', 'alias2']

        print(resolve_names_or_columns([("alias1", "column_name1"), ("alias2", "column_name2")]))
        # Outputs: ['alias1', 'alias2']

        print(resolve_names_or_columns(["column_name1", "column_name2"]))
        # Outputs: ['column_name1', 'column_name2']

    """
    if arg is None:
        return None
    if isinstance(arg, Mapping):
        return list(arg)
    if isinstance(arg, Sequence):
        if isinstance(arg[0], Tuple) and len(arg[0]) == 2:
            return [item[0] for item in arg]
        elif isinstance(arg[0], (str, Column)):
            return list(arg)
    raise ValueError("Input must be either None, a Mapping, a Sequence of Tuple, "
                     "or a Sequence of NameOrColumn.")
