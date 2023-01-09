
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
orientation(lbpugt_2,122,960). coords(lbpugt_2,189,66,960).  visible(lbpugt_2,960). happensAt(walking(lbpugt_2),960). orientation(lbpugt_4,135,960). coords(lbpugt_4,219,34,960).  visible(lbpugt_4,960). happensAt(inactive(lbpugt_4),960).
orientation(lbpugt_2,122,961). coords(lbpugt_2,190,65,961).  visible(lbpugt_2,961). happensAt(walking(lbpugt_2),961). orientation(lbpugt_4,135,961). coords(lbpugt_4,219,34,961).  visible(lbpugt_4,961). happensAt(inactive(lbpugt_4),961).
orientation(lbpugt_2,122,962). coords(lbpugt_2,190,64,962).  visible(lbpugt_2,962). happensAt(walking(lbpugt_2),962). orientation(lbpugt_4,135,962). coords(lbpugt_4,219,34,962).  visible(lbpugt_4,962). happensAt(inactive(lbpugt_4),962).
orientation(lbpugt_2,118,963). coords(lbpugt_2,189,62,963).  visible(lbpugt_2,963). happensAt(walking(lbpugt_2),963). orientation(lbpugt_4,135,963). coords(lbpugt_4,219,34,963).  visible(lbpugt_4,963). happensAt(inactive(lbpugt_4),963).
orientation(lbpugt_2,118,964). coords(lbpugt_2,190,60,964).  visible(lbpugt_2,964). happensAt(walking(lbpugt_2),964). orientation(lbpugt_4,135,964). coords(lbpugt_4,219,34,964).  visible(lbpugt_4,964). happensAt(inactive(lbpugt_4),964).
orientation(lbpugt_2,118,965). coords(lbpugt_2,190,59,965).  visible(lbpugt_2,965). happensAt(walking(lbpugt_2),965). orientation(lbpugt_4,135,965). coords(lbpugt_4,219,34,965).  visible(lbpugt_4,965). happensAt(inactive(lbpugt_4),965).
orientation(lbpugt_2,114,966). coords(lbpugt_2,190,56,966).  visible(lbpugt_2,966). happensAt(walking(lbpugt_2),966). orientation(lbpugt_4,135,966). coords(lbpugt_4,219,34,966).  visible(lbpugt_4,966). happensAt(inactive(lbpugt_4),966).
orientation(lbpugt_2,114,967). coords(lbpugt_2,190,54,967).  visible(lbpugt_2,967). happensAt(walking(lbpugt_2),967). orientation(lbpugt_4,135,967). coords(lbpugt_4,219,34,967).  visible(lbpugt_4,967). happensAt(inactive(lbpugt_4),967).
orientation(lbpugt_2,121,968). coords(lbpugt_2,190,53,968).  visible(lbpugt_2,968). happensAt(walking(lbpugt_2),968). orientation(lbpugt_4,135,968). coords(lbpugt_4,219,34,968).  visible(lbpugt_4,968). happensAt(inactive(lbpugt_4),968).
orientation(lbpugt_2,121,969). coords(lbpugt_2,190,52,969).  visible(lbpugt_2,969). happensAt(walking(lbpugt_2),969). orientation(lbpugt_4,135,969). coords(lbpugt_4,219,34,969).  visible(lbpugt_4,969). happensAt(inactive(lbpugt_4),969).
orientation(lbpugt_2,121,970). coords(lbpugt_2,190,52,970).  visible(lbpugt_2,970). happensAt(walking(lbpugt_2),970). orientation(lbpugt_4,135,970). coords(lbpugt_4,219,34,970).  visible(lbpugt_4,970). happensAt(inactive(lbpugt_4),970).
orientation(lbpugt_2,121,971). coords(lbpugt_2,190,51,971).  visible(lbpugt_2,971). happensAt(walking(lbpugt_2),971). orientation(lbpugt_4,135,971). coords(lbpugt_4,219,34,971).  visible(lbpugt_4,971). happensAt(inactive(lbpugt_4),971).
#show holdsAt/2.