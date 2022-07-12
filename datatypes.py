from typing import Any, Dict, List, Tuple, TypeVar


Genome = List[int]
Population = List[Genome]
Parents = List[Tuple[Genome, Genome]]

T = TypeVar("T")
Matrix = List[List[T]]

Solution = Tuple[Genome, int, Dict[str, List[Any]]]

History = Dict[str, List[Any]]
