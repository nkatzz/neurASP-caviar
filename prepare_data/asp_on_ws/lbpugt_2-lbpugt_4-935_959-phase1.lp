
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

	orientation(lbpugt_2,124,935). coords(lbpugt_2,201,97,935).  visible(lbpugt_2,935). happensAt(walking(lbpugt_2),935). orientation(lbpugt_4,135,935). coords(lbpugt_4,219,34,935).  visible(lbpugt_4,935). happensAt(inactive(lbpugt_4),935).
orientation(lbpugt_2,126,936). coords(lbpugt_2,199,96,936).  visible(lbpugt_2,936). happensAt(walking(lbpugt_2),936). orientation(lbpugt_4,135,936). coords(lbpugt_4,219,34,936).  visible(lbpugt_4,936). happensAt(inactive(lbpugt_4),936).
orientation(lbpugt_2,126,937). coords(lbpugt_2,197,93,937).  visible(lbpugt_2,937). happensAt(walking(lbpugt_2),937). orientation(lbpugt_4,135,937). coords(lbpugt_4,219,34,937).  visible(lbpugt_4,937). happensAt(inactive(lbpugt_4),937).
orientation(lbpugt_2,126,938). coords(lbpugt_2,196,91,938).  visible(lbpugt_2,938). happensAt(walking(lbpugt_2),938). orientation(lbpugt_4,135,938). coords(lbpugt_4,219,34,938).  visible(lbpugt_4,938). happensAt(inactive(lbpugt_4),938).
orientation(lbpugt_2,126,939). coords(lbpugt_2,195,89,939).  visible(lbpugt_2,939). happensAt(walking(lbpugt_2),939). orientation(lbpugt_4,135,939). coords(lbpugt_4,219,34,939).  visible(lbpugt_4,939). happensAt(inactive(lbpugt_4),939).
orientation(lbpugt_2,126,940). coords(lbpugt_2,195,87,940).  visible(lbpugt_2,940). happensAt(walking(lbpugt_2),940). orientation(lbpugt_4,135,940). coords(lbpugt_4,219,34,940).  visible(lbpugt_4,940). happensAt(inactive(lbpugt_4),940).
orientation(lbpugt_2,126,941). coords(lbpugt_2,195,86,941).  visible(lbpugt_2,941). happensAt(walking(lbpugt_2),941). orientation(lbpugt_4,135,941). coords(lbpugt_4,219,34,941).  visible(lbpugt_4,941). happensAt(inactive(lbpugt_4),941).
orientation(lbpugt_2,126,942). coords(lbpugt_2,194,85,942).  visible(lbpugt_2,942). happensAt(walking(lbpugt_2),942). orientation(lbpugt_4,135,942). coords(lbpugt_4,219,34,942).  visible(lbpugt_4,942). happensAt(inactive(lbpugt_4),942).
orientation(lbpugt_2,126,943). coords(lbpugt_2,194,84,943).  visible(lbpugt_2,943). happensAt(walking(lbpugt_2),943). orientation(lbpugt_4,135,943). coords(lbpugt_4,219,34,943).  visible(lbpugt_4,943). happensAt(inactive(lbpugt_4),943).
orientation(lbpugt_2,126,944). coords(lbpugt_2,194,84,944).  visible(lbpugt_2,944). happensAt(walking(lbpugt_2),944). orientation(lbpugt_4,135,944). coords(lbpugt_4,219,34,944).  visible(lbpugt_4,944). happensAt(inactive(lbpugt_4),944).
orientation(lbpugt_2,126,945). coords(lbpugt_2,194,83,945).  visible(lbpugt_2,945). happensAt(walking(lbpugt_2),945). orientation(lbpugt_4,135,945). coords(lbpugt_4,219,34,945).  visible(lbpugt_4,945). happensAt(inactive(lbpugt_4),945).
orientation(lbpugt_2,126,946). coords(lbpugt_2,194,82,946).  visible(lbpugt_2,946). happensAt(walking(lbpugt_2),946). orientation(lbpugt_4,135,946). coords(lbpugt_4,219,34,946).  visible(lbpugt_4,946). happensAt(inactive(lbpugt_4),946).
orientation(lbpugt_2,126,947). coords(lbpugt_2,194,82,947).  visible(lbpugt_2,947). happensAt(walking(lbpugt_2),947). orientation(lbpugt_4,135,947). coords(lbpugt_4,219,34,947).  visible(lbpugt_4,947). happensAt(inactive(lbpugt_4),947).
orientation(lbpugt_2,126,948). coords(lbpugt_2,193,81,948).  visible(lbpugt_2,948). happensAt(walking(lbpugt_2),948). orientation(lbpugt_4,135,948). coords(lbpugt_4,219,34,948).  visible(lbpugt_4,948). happensAt(inactive(lbpugt_4),948).
orientation(lbpugt_2,126,949). coords(lbpugt_2,193,80,949).  visible(lbpugt_2,949). happensAt(walking(lbpugt_2),949). orientation(lbpugt_4,135,949). coords(lbpugt_4,219,34,949).  visible(lbpugt_4,949). happensAt(inactive(lbpugt_4),949).
orientation(lbpugt_2,126,950). coords(lbpugt_2,192,78,950).  visible(lbpugt_2,950). happensAt(walking(lbpugt_2),950). orientation(lbpugt_4,135,950). coords(lbpugt_4,219,34,950).  visible(lbpugt_4,950). happensAt(inactive(lbpugt_4),950).
orientation(lbpugt_2,126,951). coords(lbpugt_2,192,76,951).  visible(lbpugt_2,951). happensAt(walking(lbpugt_2),951). orientation(lbpugt_4,135,951). coords(lbpugt_4,219,34,951).  visible(lbpugt_4,951). happensAt(inactive(lbpugt_4),951).
orientation(lbpugt_2,126,952). coords(lbpugt_2,191,74,952).  visible(lbpugt_2,952). happensAt(walking(lbpugt_2),952). orientation(lbpugt_4,135,952). coords(lbpugt_4,219,34,952).  visible(lbpugt_4,952). happensAt(inactive(lbpugt_4),952).
orientation(lbpugt_2,131,953). coords(lbpugt_2,191,71,953).  visible(lbpugt_2,953). happensAt(walking(lbpugt_2),953). orientation(lbpugt_4,135,953). coords(lbpugt_4,219,34,953).  visible(lbpugt_4,953). happensAt(inactive(lbpugt_4),953).
orientation(lbpugt_2,131,954). coords(lbpugt_2,191,69,954).  visible(lbpugt_2,954). happensAt(walking(lbpugt_2),954). orientation(lbpugt_4,135,954). coords(lbpugt_4,219,34,954).  visible(lbpugt_4,954). happensAt(inactive(lbpugt_4),954).
orientation(lbpugt_2,131,955). coords(lbpugt_2,191,69,955).  visible(lbpugt_2,955). happensAt(walking(lbpugt_2),955). orientation(lbpugt_4,135,955). coords(lbpugt_4,219,34,955).  visible(lbpugt_4,955). happensAt(inactive(lbpugt_4),955).
orientation(lbpugt_2,131,956). coords(lbpugt_2,189,68,956).  visible(lbpugt_2,956). happensAt(walking(lbpugt_2),956). orientation(lbpugt_4,135,956). coords(lbpugt_4,219,34,956).  visible(lbpugt_4,956). happensAt(inactive(lbpugt_4),956).
orientation(lbpugt_2,131,957). coords(lbpugt_2,189,67,957).  visible(lbpugt_2,957). happensAt(walking(lbpugt_2),957). orientation(lbpugt_4,135,957). coords(lbpugt_4,219,34,957).  visible(lbpugt_4,957). happensAt(inactive(lbpugt_4),957).
orientation(lbpugt_2,131,958). coords(lbpugt_2,189,67,958).  visible(lbpugt_2,958). happensAt(walking(lbpugt_2),958). orientation(lbpugt_4,135,958). coords(lbpugt_4,219,34,958).  visible(lbpugt_4,958). happensAt(inactive(lbpugt_4),958).
orientation(lbpugt_2,122,959). coords(lbpugt_2,189,66,959).  visible(lbpugt_2,959). happensAt(walking(lbpugt_2),959). orientation(lbpugt_4,135,959). coords(lbpugt_4,219,34,959).  visible(lbpugt_4,959). happensAt(inactive(lbpugt_4),959).
#show holdsAt/2.