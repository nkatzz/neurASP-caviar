#script (python)
import math
import clingo
from clingo import Number

def eucldist(x1,y1,x2,y2):
    p1 = [x1.number,y1.number]
    p2 = [x2.number,y2.number]
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return Number(int(distance))    
#end.

dist(Id1,Id2,T,Ypot):-
       coords(Id1,X1,Y1,T),
       coords(Id2,X2,Y2,T),
       Id1 != Id2,
       X = X1-X2,
       Y = Y1-Y2,
       Ypot = @eucldist(X1,Y1,X2,Y2).

close(Id1,Id2,Threshold,Time):-
       dist(Id1,Id2,Time,Distance),
       Id1 != Id2,
       threshold_value(Threshold),
       Distance <= Threshold,
       person(Id1),person(Id2),time(Time).

far(Id1,Id2,Threshold,Time):-
       dist(Id1,Id2,Time,Distance),
       Id1 != Id2,
       threshold_value(Threshold),
       Distance > Threshold,
       person(Id1),person(Id2),time(Time).

orientationMove(Id1,Id2,Time):-
       Id1 != Id2,
       orientation(Id1,X,Time),
       orientation(Id2,Y,Time),
       Diff = |X-Y|,
       orientation_threshold(Threshold),
       Diff <= Threshold,
       person(Id1),person(Id2),time(Time).

orientationFar(Id1,Id2,Time):-
       Id1 != Id2,
       orientation(Id1,X,Time),
       orientation(Id2,Y,Time),
       Diff = |X-Y|,
       orientation_threshold(Threshold),
       Diff > Threshold,
       person(Id1),person(Id2),time(Time).

orientation_threshold(45).
threshold_value(25).
threshold_value(24).
threshold_value(27).
threshold_value(34).
threshold_value(40).

