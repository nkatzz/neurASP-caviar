#script (python)
from clingo import Number, Tuple_
times = []
def collect_all(a):
    times.append(a)
    return Number(1)
def sort():
    return [Tuple_((Number(i), val)) for i, val in enumerate(sorted(times))]
#end.

% ASP code.
collect_all.
collect_all :- time(X), @collect_all(X) == 0.
sorted_pair(X,N) :- collect_all, (X,N) = @sort().
next(X, Y) :- sorted_pair(A,X), sorted_pair(A+1,Y).

