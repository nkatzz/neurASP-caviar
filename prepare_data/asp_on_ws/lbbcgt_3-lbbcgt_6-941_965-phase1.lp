
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

	orientation(lbbcgt_3,163,941). coords(lbbcgt_3,35,180,941).  visible(lbbcgt_3,941). happensAt(walking(lbbcgt_3),941). orientation(lbbcgt_6,147,941). coords(lbbcgt_6,90,84,941).  visible(lbbcgt_6,941). happensAt(walking(lbbcgt_6),941).
orientation(lbbcgt_3,163,942). coords(lbbcgt_3,35,180,942).  visible(lbbcgt_3,942). happensAt(walking(lbbcgt_3),942). orientation(lbbcgt_6,144,942). coords(lbbcgt_6,89,85,942).  visible(lbbcgt_6,942). happensAt(walking(lbbcgt_6),942).
orientation(lbbcgt_3,163,943). coords(lbbcgt_3,35,181,943).  visible(lbbcgt_3,943). happensAt(walking(lbbcgt_3),943). orientation(lbbcgt_6,143,943). coords(lbbcgt_6,89,84,943).  visible(lbbcgt_6,943). happensAt(walking(lbbcgt_6),943).
orientation(lbbcgt_3,163,944). coords(lbbcgt_3,35,182,944).  visible(lbbcgt_3,944). happensAt(walking(lbbcgt_3),944). orientation(lbbcgt_6,143,944). coords(lbbcgt_6,89,85,944).  visible(lbbcgt_6,944). happensAt(walking(lbbcgt_6),944).
orientation(lbbcgt_3,163,945). coords(lbbcgt_3,35,184,945).  visible(lbbcgt_3,945). happensAt(walking(lbbcgt_3),945). orientation(lbbcgt_6,143,945). coords(lbbcgt_6,88,85,945).  visible(lbbcgt_6,945). happensAt(walking(lbbcgt_6),945).
orientation(lbbcgt_3,163,946). coords(lbbcgt_3,36,185,946).  visible(lbbcgt_3,946). happensAt(walking(lbbcgt_3),946). orientation(lbbcgt_6,145,946). coords(lbbcgt_6,89,86,946).  visible(lbbcgt_6,946). happensAt(walking(lbbcgt_6),946).
orientation(lbbcgt_3,163,947). coords(lbbcgt_3,36,186,947).  visible(lbbcgt_3,947). happensAt(walking(lbbcgt_3),947). orientation(lbbcgt_6,139,947). coords(lbbcgt_6,88,86,947).  visible(lbbcgt_6,947). happensAt(walking(lbbcgt_6),947).
orientation(lbbcgt_3,163,948). coords(lbbcgt_3,36,187,948).  visible(lbbcgt_3,948). happensAt(walking(lbbcgt_3),948). orientation(lbbcgt_6,144,948). coords(lbbcgt_6,88,86,948).  visible(lbbcgt_6,948). happensAt(walking(lbbcgt_6),948).
orientation(lbbcgt_3,163,949). coords(lbbcgt_3,36,188,949).  visible(lbbcgt_3,949). happensAt(walking(lbbcgt_3),949). orientation(lbbcgt_6,144,949). coords(lbbcgt_6,88,86,949).  visible(lbbcgt_6,949). happensAt(walking(lbbcgt_6),949).
orientation(lbbcgt_3,163,950). coords(lbbcgt_3,36,188,950).  visible(lbbcgt_3,950). happensAt(walking(lbbcgt_3),950). orientation(lbbcgt_6,148,950). coords(lbbcgt_6,87,86,950).  visible(lbbcgt_6,950). happensAt(walking(lbbcgt_6),950).
orientation(lbbcgt_3,163,951). coords(lbbcgt_3,36,189,951).  visible(lbbcgt_3,951). happensAt(walking(lbbcgt_3),951). orientation(lbbcgt_6,151,951). coords(lbbcgt_6,87,86,951).  visible(lbbcgt_6,951). happensAt(walking(lbbcgt_6),951).
orientation(lbbcgt_3,163,952). coords(lbbcgt_3,36,189,952).  visible(lbbcgt_3,952). happensAt(walking(lbbcgt_3),952). orientation(lbbcgt_6,151,952). coords(lbbcgt_6,86,86,952).  visible(lbbcgt_6,952). happensAt(walking(lbbcgt_6),952).
orientation(lbbcgt_3,163,953). coords(lbbcgt_3,35,190,953).  visible(lbbcgt_3,953). happensAt(walking(lbbcgt_3),953). orientation(lbbcgt_6,151,953). coords(lbbcgt_6,85,86,953).  visible(lbbcgt_6,953). happensAt(walking(lbbcgt_6),953).
orientation(lbbcgt_3,163,954). coords(lbbcgt_3,35,190,954).  visible(lbbcgt_3,954). happensAt(walking(lbbcgt_3),954). orientation(lbbcgt_6,151,954). coords(lbbcgt_6,84,86,954).  visible(lbbcgt_6,954). happensAt(walking(lbbcgt_6),954).
orientation(lbbcgt_3,163,955). coords(lbbcgt_3,35,190,955).  visible(lbbcgt_3,955). happensAt(walking(lbbcgt_3),955). orientation(lbbcgt_6,151,955). coords(lbbcgt_6,83,86,955).  visible(lbbcgt_6,955). happensAt(walking(lbbcgt_6),955).
orientation(lbbcgt_3,163,956). coords(lbbcgt_3,35,191,956).  visible(lbbcgt_3,956). happensAt(walking(lbbcgt_3),956). orientation(lbbcgt_6,151,956). coords(lbbcgt_6,82,86,956).  visible(lbbcgt_6,956). happensAt(walking(lbbcgt_6),956).
orientation(lbbcgt_3,163,957). coords(lbbcgt_3,35,191,957).  visible(lbbcgt_3,957). happensAt(walking(lbbcgt_3),957). orientation(lbbcgt_6,151,957). coords(lbbcgt_6,81,86,957).  visible(lbbcgt_6,957). happensAt(walking(lbbcgt_6),957).
orientation(lbbcgt_3,163,958). coords(lbbcgt_3,35,192,958).  visible(lbbcgt_3,958). happensAt(walking(lbbcgt_3),958). orientation(lbbcgt_6,151,958). coords(lbbcgt_6,81,86,958).  visible(lbbcgt_6,958). happensAt(walking(lbbcgt_6),958).
orientation(lbbcgt_3,163,959). coords(lbbcgt_3,35,193,959).  visible(lbbcgt_3,959). happensAt(walking(lbbcgt_3),959). orientation(lbbcgt_6,151,959). coords(lbbcgt_6,81,86,959).  visible(lbbcgt_6,959). happensAt(walking(lbbcgt_6),959).
orientation(lbbcgt_3,163,960). coords(lbbcgt_3,35,195,960).  visible(lbbcgt_3,960). happensAt(walking(lbbcgt_3),960). orientation(lbbcgt_6,151,960). coords(lbbcgt_6,81,87,960).  visible(lbbcgt_6,960). happensAt(walking(lbbcgt_6),960).
orientation(lbbcgt_3,163,961). coords(lbbcgt_3,35,197,961).  visible(lbbcgt_3,961). happensAt(walking(lbbcgt_3),961). orientation(lbbcgt_6,151,961). coords(lbbcgt_6,81,87,961).  visible(lbbcgt_6,961). happensAt(walking(lbbcgt_6),961).
orientation(lbbcgt_3,163,962). coords(lbbcgt_3,35,197,962).  visible(lbbcgt_3,962). happensAt(walking(lbbcgt_3),962). orientation(lbbcgt_6,151,962). coords(lbbcgt_6,81,87,962).  visible(lbbcgt_6,962). happensAt(walking(lbbcgt_6),962).
orientation(lbbcgt_3,163,963). coords(lbbcgt_3,35,198,963).  visible(lbbcgt_3,963). happensAt(walking(lbbcgt_3),963). orientation(lbbcgt_6,151,963). coords(lbbcgt_6,81,87,963).  visible(lbbcgt_6,963). happensAt(walking(lbbcgt_6),963).
orientation(lbbcgt_3,163,964). coords(lbbcgt_3,35,198,964).  visible(lbbcgt_3,964). happensAt(walking(lbbcgt_3),964). orientation(lbbcgt_6,151,964). coords(lbbcgt_6,81,87,964).  visible(lbbcgt_6,964). happensAt(walking(lbbcgt_6),964).
orientation(lbbcgt_3,163,965). coords(lbbcgt_3,35,199,965).  visible(lbbcgt_3,965). happensAt(walking(lbbcgt_3),965). orientation(lbbcgt_6,151,965). coords(lbbcgt_6,79,87,965).  visible(lbbcgt_6,965). happensAt(walking(lbbcgt_6),965).
#show holdsAt/2.