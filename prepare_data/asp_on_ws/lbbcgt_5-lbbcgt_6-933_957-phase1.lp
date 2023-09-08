
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbbcgt_5,158,933). coords(lbbcgt_5,203,187,933).  visible(lbbcgt_5,933). happensAt(walking(lbbcgt_5),933). orientation(lbbcgt_6,144,933). coords(lbbcgt_6,96,83,933).  visible(lbbcgt_6,933). happensAt(walking(lbbcgt_6),933).
orientation(lbbcgt_5,158,934). coords(lbbcgt_5,203,186,934).  visible(lbbcgt_5,934). happensAt(walking(lbbcgt_5),934). orientation(lbbcgt_6,144,934). coords(lbbcgt_6,96,83,934).  visible(lbbcgt_6,934). happensAt(walking(lbbcgt_6),934).
orientation(lbbcgt_5,158,935). coords(lbbcgt_5,203,185,935).  visible(lbbcgt_5,935). happensAt(walking(lbbcgt_5),935). orientation(lbbcgt_6,150,935). coords(lbbcgt_6,96,83,935).  visible(lbbcgt_6,935). happensAt(walking(lbbcgt_6),935).
orientation(lbbcgt_5,158,936). coords(lbbcgt_5,202,184,936).  visible(lbbcgt_5,936). happensAt(walking(lbbcgt_5),936). orientation(lbbcgt_6,150,936). coords(lbbcgt_6,95,84,936).  visible(lbbcgt_6,936). happensAt(walking(lbbcgt_6),936).
orientation(lbbcgt_5,158,937). coords(lbbcgt_5,202,183,937).  visible(lbbcgt_5,937). happensAt(walking(lbbcgt_5),937). orientation(lbbcgt_6,150,937). coords(lbbcgt_6,95,84,937).  visible(lbbcgt_6,937). happensAt(walking(lbbcgt_6),937).
orientation(lbbcgt_5,158,938). coords(lbbcgt_5,201,182,938).  visible(lbbcgt_5,938). happensAt(walking(lbbcgt_5),938). orientation(lbbcgt_6,150,938). coords(lbbcgt_6,94,84,938).  visible(lbbcgt_6,938). happensAt(walking(lbbcgt_6),938).
orientation(lbbcgt_5,158,939). coords(lbbcgt_5,201,180,939).  visible(lbbcgt_5,939). happensAt(walking(lbbcgt_5),939). orientation(lbbcgt_6,150,939). coords(lbbcgt_6,94,85,939).  visible(lbbcgt_6,939). happensAt(walking(lbbcgt_6),939).
orientation(lbbcgt_5,158,940). coords(lbbcgt_5,199,177,940).  visible(lbbcgt_5,940). happensAt(walking(lbbcgt_5),940). orientation(lbbcgt_6,150,940). coords(lbbcgt_6,92,84,940).  visible(lbbcgt_6,940). happensAt(walking(lbbcgt_6),940).
orientation(lbbcgt_5,158,941). coords(lbbcgt_5,198,175,941).  visible(lbbcgt_5,941). happensAt(walking(lbbcgt_5),941). orientation(lbbcgt_6,147,941). coords(lbbcgt_6,90,84,941).  visible(lbbcgt_6,941). happensAt(walking(lbbcgt_6),941).
orientation(lbbcgt_5,158,942). coords(lbbcgt_5,197,173,942).  visible(lbbcgt_5,942). happensAt(walking(lbbcgt_5),942). orientation(lbbcgt_6,144,942). coords(lbbcgt_6,89,85,942).  visible(lbbcgt_6,942). happensAt(walking(lbbcgt_6),942).
orientation(lbbcgt_5,158,943). coords(lbbcgt_5,195,172,943).  visible(lbbcgt_5,943). happensAt(walking(lbbcgt_5),943). orientation(lbbcgt_6,143,943). coords(lbbcgt_6,89,84,943).  visible(lbbcgt_6,943). happensAt(walking(lbbcgt_6),943).
orientation(lbbcgt_5,158,944). coords(lbbcgt_5,193,171,944).  visible(lbbcgt_5,944). happensAt(walking(lbbcgt_5),944). orientation(lbbcgt_6,143,944). coords(lbbcgt_6,89,85,944).  visible(lbbcgt_6,944). happensAt(walking(lbbcgt_6),944).
orientation(lbbcgt_5,158,945). coords(lbbcgt_5,192,169,945).  visible(lbbcgt_5,945). happensAt(walking(lbbcgt_5),945). orientation(lbbcgt_6,143,945). coords(lbbcgt_6,88,85,945).  visible(lbbcgt_6,945). happensAt(walking(lbbcgt_6),945).
orientation(lbbcgt_5,158,946). coords(lbbcgt_5,191,169,946).  visible(lbbcgt_5,946). happensAt(walking(lbbcgt_5),946). orientation(lbbcgt_6,145,946). coords(lbbcgt_6,89,86,946).  visible(lbbcgt_6,946). happensAt(walking(lbbcgt_6),946).
orientation(lbbcgt_5,158,947). coords(lbbcgt_5,191,168,947).  visible(lbbcgt_5,947). happensAt(walking(lbbcgt_5),947). orientation(lbbcgt_6,139,947). coords(lbbcgt_6,88,86,947).  visible(lbbcgt_6,947). happensAt(walking(lbbcgt_6),947).
orientation(lbbcgt_5,158,948). coords(lbbcgt_5,191,166,948).  visible(lbbcgt_5,948). happensAt(walking(lbbcgt_5),948). orientation(lbbcgt_6,144,948). coords(lbbcgt_6,88,86,948).  visible(lbbcgt_6,948). happensAt(walking(lbbcgt_6),948).
orientation(lbbcgt_5,158,949). coords(lbbcgt_5,191,166,949).  visible(lbbcgt_5,949). happensAt(walking(lbbcgt_5),949). orientation(lbbcgt_6,144,949). coords(lbbcgt_6,88,86,949).  visible(lbbcgt_6,949). happensAt(walking(lbbcgt_6),949).
orientation(lbbcgt_5,158,950). coords(lbbcgt_5,191,163,950).  visible(lbbcgt_5,950). happensAt(walking(lbbcgt_5),950). orientation(lbbcgt_6,148,950). coords(lbbcgt_6,87,86,950).  visible(lbbcgt_6,950). happensAt(walking(lbbcgt_6),950).
orientation(lbbcgt_5,158,951). coords(lbbcgt_5,190,162,951).  visible(lbbcgt_5,951). happensAt(walking(lbbcgt_5),951). orientation(lbbcgt_6,151,951). coords(lbbcgt_6,87,86,951).  visible(lbbcgt_6,951). happensAt(walking(lbbcgt_6),951).
orientation(lbbcgt_5,158,952). coords(lbbcgt_5,190,161,952).  visible(lbbcgt_5,952). happensAt(walking(lbbcgt_5),952). orientation(lbbcgt_6,151,952). coords(lbbcgt_6,86,86,952).  visible(lbbcgt_6,952). happensAt(walking(lbbcgt_6),952).
orientation(lbbcgt_5,158,953). coords(lbbcgt_5,189,160,953).  visible(lbbcgt_5,953). happensAt(walking(lbbcgt_5),953). orientation(lbbcgt_6,151,953). coords(lbbcgt_6,85,86,953).  visible(lbbcgt_6,953). happensAt(walking(lbbcgt_6),953).
orientation(lbbcgt_5,158,954). coords(lbbcgt_5,189,160,954).  visible(lbbcgt_5,954). happensAt(walking(lbbcgt_5),954). orientation(lbbcgt_6,151,954). coords(lbbcgt_6,84,86,954).  visible(lbbcgt_6,954). happensAt(walking(lbbcgt_6),954).
orientation(lbbcgt_5,158,955). coords(lbbcgt_5,188,158,955).  visible(lbbcgt_5,955). happensAt(walking(lbbcgt_5),955). orientation(lbbcgt_6,151,955). coords(lbbcgt_6,83,86,955).  visible(lbbcgt_6,955). happensAt(walking(lbbcgt_6),955).
orientation(lbbcgt_5,158,956). coords(lbbcgt_5,188,156,956).  visible(lbbcgt_5,956). happensAt(walking(lbbcgt_5),956). orientation(lbbcgt_6,151,956). coords(lbbcgt_6,82,86,956).  visible(lbbcgt_6,956). happensAt(walking(lbbcgt_6),956).
orientation(lbbcgt_5,158,957). coords(lbbcgt_5,188,155,957).  visible(lbbcgt_5,957). happensAt(walking(lbbcgt_5),957). orientation(lbbcgt_6,151,957). coords(lbbcgt_6,81,86,957).  visible(lbbcgt_6,957). happensAt(walking(lbbcgt_6),957).
#show holdsAt/2.