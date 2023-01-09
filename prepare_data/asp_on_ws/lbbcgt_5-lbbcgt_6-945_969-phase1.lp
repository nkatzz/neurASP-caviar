
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
orientation(lbbcgt_5,158,958). coords(lbbcgt_5,187,154,958).  visible(lbbcgt_5,958). happensAt(walking(lbbcgt_5),958). orientation(lbbcgt_6,151,958). coords(lbbcgt_6,81,86,958).  visible(lbbcgt_6,958). happensAt(walking(lbbcgt_6),958).
orientation(lbbcgt_5,158,959). coords(lbbcgt_5,187,153,959).  visible(lbbcgt_5,959). happensAt(walking(lbbcgt_5),959). orientation(lbbcgt_6,151,959). coords(lbbcgt_6,81,86,959).  visible(lbbcgt_6,959). happensAt(walking(lbbcgt_6),959).
orientation(lbbcgt_5,158,960). coords(lbbcgt_5,187,151,960).  visible(lbbcgt_5,960). happensAt(walking(lbbcgt_5),960). orientation(lbbcgt_6,151,960). coords(lbbcgt_6,81,87,960).  visible(lbbcgt_6,960). happensAt(walking(lbbcgt_6),960).
orientation(lbbcgt_5,158,961). coords(lbbcgt_5,186,150,961).  visible(lbbcgt_5,961). happensAt(walking(lbbcgt_5),961). orientation(lbbcgt_6,151,961). coords(lbbcgt_6,81,87,961).  visible(lbbcgt_6,961). happensAt(walking(lbbcgt_6),961).
orientation(lbbcgt_5,158,962). coords(lbbcgt_5,186,150,962).  visible(lbbcgt_5,962). happensAt(walking(lbbcgt_5),962). orientation(lbbcgt_6,151,962). coords(lbbcgt_6,81,87,962).  visible(lbbcgt_6,962). happensAt(walking(lbbcgt_6),962).
orientation(lbbcgt_5,158,963). coords(lbbcgt_5,186,148,963).  visible(lbbcgt_5,963). happensAt(walking(lbbcgt_5),963). orientation(lbbcgt_6,151,963). coords(lbbcgt_6,81,87,963).  visible(lbbcgt_6,963). happensAt(walking(lbbcgt_6),963).
orientation(lbbcgt_5,158,964). coords(lbbcgt_5,186,147,964).  visible(lbbcgt_5,964). happensAt(walking(lbbcgt_5),964). orientation(lbbcgt_6,151,964). coords(lbbcgt_6,81,87,964).  visible(lbbcgt_6,964). happensAt(walking(lbbcgt_6),964).
orientation(lbbcgt_5,158,965). coords(lbbcgt_5,186,144,965).  visible(lbbcgt_5,965). happensAt(walking(lbbcgt_5),965). orientation(lbbcgt_6,151,965). coords(lbbcgt_6,79,87,965).  visible(lbbcgt_6,965). happensAt(walking(lbbcgt_6),965).
orientation(lbbcgt_5,158,966). coords(lbbcgt_5,187,143,966).  visible(lbbcgt_5,966). happensAt(walking(lbbcgt_5),966). orientation(lbbcgt_6,151,966). coords(lbbcgt_6,79,86,966).  visible(lbbcgt_6,966). happensAt(walking(lbbcgt_6),966).
orientation(lbbcgt_5,158,967). coords(lbbcgt_5,187,142,967).  visible(lbbcgt_5,967). happensAt(walking(lbbcgt_5),967). orientation(lbbcgt_6,151,967). coords(lbbcgt_6,78,87,967).  visible(lbbcgt_6,967). happensAt(walking(lbbcgt_6),967).
orientation(lbbcgt_5,154,968). coords(lbbcgt_5,187,142,968).  visible(lbbcgt_5,968). happensAt(walking(lbbcgt_5),968). orientation(lbbcgt_6,151,968). coords(lbbcgt_6,78,87,968).  visible(lbbcgt_6,968). happensAt(walking(lbbcgt_6),968).
orientation(lbbcgt_5,154,969). coords(lbbcgt_5,187,142,969).  visible(lbbcgt_5,969). happensAt(walking(lbbcgt_5),969). orientation(lbbcgt_6,151,969). coords(lbbcgt_6,77,87,969).  visible(lbbcgt_6,969). happensAt(walking(lbbcgt_6),969).
#show holdsAt/2.