
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

	orientation(wk2gt_1,133,944). coords(wk2gt_1,121,48,944).  visible(wk2gt_1,944). happensAt(walking(wk2gt_1),944). orientation(wk2gt_5,150,944). coords(wk2gt_5,76,99,944).  visible(wk2gt_5,944). happensAt(walking(wk2gt_5),944).
orientation(wk2gt_1,133,945). coords(wk2gt_1,121,48,945).  visible(wk2gt_1,945). happensAt(walking(wk2gt_1),945). orientation(wk2gt_5,150,945). coords(wk2gt_5,76,99,945).  visible(wk2gt_5,945). happensAt(walking(wk2gt_5),945).
orientation(wk2gt_1,133,946). coords(wk2gt_1,121,48,946).  visible(wk2gt_1,946). happensAt(walking(wk2gt_1),946). orientation(wk2gt_5,150,946). coords(wk2gt_5,76,99,946).  visible(wk2gt_5,946). happensAt(walking(wk2gt_5),946).
orientation(wk2gt_1,133,947). coords(wk2gt_1,121,48,947).  visible(wk2gt_1,947). happensAt(walking(wk2gt_1),947). orientation(wk2gt_5,150,947). coords(wk2gt_5,76,99,947).  visible(wk2gt_5,947). happensAt(walking(wk2gt_5),947).
orientation(wk2gt_1,133,948). coords(wk2gt_1,121,48,948).  visible(wk2gt_1,948). happensAt(walking(wk2gt_1),948). orientation(wk2gt_5,150,948). coords(wk2gt_5,76,100,948).  visible(wk2gt_5,948). happensAt(walking(wk2gt_5),948).
orientation(wk2gt_1,133,949). coords(wk2gt_1,121,48,949).  visible(wk2gt_1,949). happensAt(walking(wk2gt_1),949). orientation(wk2gt_5,150,949). coords(wk2gt_5,76,100,949).  visible(wk2gt_5,949). happensAt(walking(wk2gt_5),949).
orientation(wk2gt_1,133,950). coords(wk2gt_1,121,48,950).  visible(wk2gt_1,950). happensAt(walking(wk2gt_1),950). orientation(wk2gt_5,150,950). coords(wk2gt_5,77,100,950).  visible(wk2gt_5,950). happensAt(walking(wk2gt_5),950).
orientation(wk2gt_1,133,951). coords(wk2gt_1,122,47,951).  visible(wk2gt_1,951). happensAt(walking(wk2gt_1),951). orientation(wk2gt_5,150,951). coords(wk2gt_5,78,100,951).  visible(wk2gt_5,951). happensAt(walking(wk2gt_5),951).
orientation(wk2gt_1,133,952). coords(wk2gt_1,122,47,952).  visible(wk2gt_1,952). happensAt(walking(wk2gt_1),952). orientation(wk2gt_5,150,952). coords(wk2gt_5,79,100,952).  visible(wk2gt_5,952). happensAt(walking(wk2gt_5),952).
orientation(wk2gt_1,133,953). coords(wk2gt_1,123,48,953).  visible(wk2gt_1,953). happensAt(walking(wk2gt_1),953). orientation(wk2gt_5,150,953). coords(wk2gt_5,79,100,953).  visible(wk2gt_5,953). happensAt(walking(wk2gt_5),953).
orientation(wk2gt_1,133,954). coords(wk2gt_1,123,48,954).  visible(wk2gt_1,954). happensAt(walking(wk2gt_1),954). orientation(wk2gt_5,150,954). coords(wk2gt_5,80,100,954).  visible(wk2gt_5,954). happensAt(walking(wk2gt_5),954).
orientation(wk2gt_1,133,955). coords(wk2gt_1,123,47,955).  visible(wk2gt_1,955). happensAt(walking(wk2gt_1),955). orientation(wk2gt_5,150,955). coords(wk2gt_5,80,100,955).  visible(wk2gt_5,955). happensAt(walking(wk2gt_5),955).
orientation(wk2gt_1,133,956). coords(wk2gt_1,123,46,956).  visible(wk2gt_1,956). happensAt(walking(wk2gt_1),956). orientation(wk2gt_5,150,956). coords(wk2gt_5,81,100,956).  visible(wk2gt_5,956). happensAt(walking(wk2gt_5),956).
orientation(wk2gt_1,133,957). coords(wk2gt_1,123,46,957).  visible(wk2gt_1,957). happensAt(walking(wk2gt_1),957). orientation(wk2gt_5,150,957). coords(wk2gt_5,82,101,957).  visible(wk2gt_5,957). happensAt(walking(wk2gt_5),957).
orientation(wk2gt_1,133,958). coords(wk2gt_1,124,46,958).  visible(wk2gt_1,958). happensAt(walking(wk2gt_1),958). orientation(wk2gt_5,150,958). coords(wk2gt_5,82,101,958).  visible(wk2gt_5,958). happensAt(walking(wk2gt_5),958).
orientation(wk2gt_1,133,959). coords(wk2gt_1,124,46,959).  visible(wk2gt_1,959). happensAt(walking(wk2gt_1),959). orientation(wk2gt_5,150,959). coords(wk2gt_5,82,101,959).  visible(wk2gt_5,959). happensAt(walking(wk2gt_5),959).
orientation(wk2gt_1,133,960). coords(wk2gt_1,125,45,960).  visible(wk2gt_1,960). happensAt(walking(wk2gt_1),960). orientation(wk2gt_5,150,960). coords(wk2gt_5,82,101,960).  visible(wk2gt_5,960). happensAt(walking(wk2gt_5),960).
orientation(wk2gt_1,133,961). coords(wk2gt_1,125,45,961).  visible(wk2gt_1,961). happensAt(walking(wk2gt_1),961). orientation(wk2gt_5,150,961). coords(wk2gt_5,82,102,961).  visible(wk2gt_5,961). happensAt(walking(wk2gt_5),961).
orientation(wk2gt_1,133,962). coords(wk2gt_1,125,45,962).  visible(wk2gt_1,962). happensAt(walking(wk2gt_1),962). orientation(wk2gt_5,150,962). coords(wk2gt_5,82,103,962).  visible(wk2gt_5,962). happensAt(walking(wk2gt_5),962).
orientation(wk2gt_1,133,963). coords(wk2gt_1,125,45,963).  visible(wk2gt_1,963). happensAt(walking(wk2gt_1),963). orientation(wk2gt_5,150,963). coords(wk2gt_5,82,104,963).  visible(wk2gt_5,963). happensAt(walking(wk2gt_5),963).
orientation(wk2gt_1,133,964). coords(wk2gt_1,125,45,964).  visible(wk2gt_1,964). happensAt(walking(wk2gt_1),964). orientation(wk2gt_5,150,964). coords(wk2gt_5,83,104,964).  visible(wk2gt_5,964). happensAt(walking(wk2gt_5),964).
orientation(wk2gt_1,133,965). coords(wk2gt_1,126,45,965).  visible(wk2gt_1,965). happensAt(walking(wk2gt_1),965). orientation(wk2gt_5,150,965). coords(wk2gt_5,84,105,965).  visible(wk2gt_5,965). happensAt(walking(wk2gt_5),965).
orientation(wk2gt_1,133,966). coords(wk2gt_1,127,45,966).  visible(wk2gt_1,966). happensAt(walking(wk2gt_1),966). orientation(wk2gt_5,150,966). coords(wk2gt_5,85,106,966).  visible(wk2gt_5,966). happensAt(walking(wk2gt_5),966).
orientation(wk2gt_1,133,967). coords(wk2gt_1,127,45,967).  visible(wk2gt_1,967). happensAt(walking(wk2gt_1),967). orientation(wk2gt_5,150,967). coords(wk2gt_5,85,106,967).  visible(wk2gt_5,967). happensAt(walking(wk2gt_5),967).
orientation(wk2gt_1,133,968). coords(wk2gt_1,127,45,968).  visible(wk2gt_1,968). happensAt(walking(wk2gt_1),968). orientation(wk2gt_5,150,968). coords(wk2gt_5,86,106,968).  visible(wk2gt_5,968). happensAt(walking(wk2gt_5),968).
#show holdsAt/2.