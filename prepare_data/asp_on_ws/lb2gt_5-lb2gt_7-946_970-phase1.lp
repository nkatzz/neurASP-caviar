
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

	orientation(lb2gt_5,152,946). coords(lb2gt_5,87,118,946).  visible(lb2gt_5,946). happensAt(walking(lb2gt_5),946). orientation(lb2gt_7,0,946). coords(lb2gt_7,110,267,946).  happensAt(appear(lb2gt_7),946). happensAt(walking(lb2gt_7),946).
orientation(lb2gt_5,152,947). coords(lb2gt_5,88,119,947).  visible(lb2gt_5,947). happensAt(walking(lb2gt_5),947). orientation(lb2gt_7,0,947). coords(lb2gt_7,110,267,947).  visible(lb2gt_7,947). happensAt(walking(lb2gt_7),947).
orientation(lb2gt_5,152,948). coords(lb2gt_5,89,119,948).  visible(lb2gt_5,948). happensAt(walking(lb2gt_5),948). orientation(lb2gt_7,0,948). coords(lb2gt_7,110,267,948).  visible(lb2gt_7,948). happensAt(walking(lb2gt_7),948).
orientation(lb2gt_5,152,949). coords(lb2gt_5,90,120,949).  visible(lb2gt_5,949). happensAt(walking(lb2gt_5),949). orientation(lb2gt_7,0,949). coords(lb2gt_7,110,266,949).  visible(lb2gt_7,949). happensAt(walking(lb2gt_7),949).
orientation(lb2gt_5,152,950). coords(lb2gt_5,91,120,950).  visible(lb2gt_5,950). happensAt(walking(lb2gt_5),950). orientation(lb2gt_7,0,950). coords(lb2gt_7,110,266,950).  visible(lb2gt_7,950). happensAt(walking(lb2gt_7),950).
orientation(lb2gt_5,152,951). coords(lb2gt_5,91,120,951).  visible(lb2gt_5,951). happensAt(walking(lb2gt_5),951). orientation(lb2gt_7,0,951). coords(lb2gt_7,110,266,951).  visible(lb2gt_7,951). happensAt(walking(lb2gt_7),951).
orientation(lb2gt_5,152,952). coords(lb2gt_5,92,121,952).  visible(lb2gt_5,952). happensAt(walking(lb2gt_5),952). orientation(lb2gt_7,0,952). coords(lb2gt_7,110,266,952).  visible(lb2gt_7,952). happensAt(walking(lb2gt_7),952).
orientation(lb2gt_5,152,953). coords(lb2gt_5,92,121,953).  visible(lb2gt_5,953). happensAt(walking(lb2gt_5),953). orientation(lb2gt_7,0,953). coords(lb2gt_7,110,266,953).  visible(lb2gt_7,953). happensAt(walking(lb2gt_7),953).
orientation(lb2gt_5,152,954). coords(lb2gt_5,92,121,954).  visible(lb2gt_5,954). happensAt(walking(lb2gt_5),954). orientation(lb2gt_7,0,954). coords(lb2gt_7,110,265,954).  visible(lb2gt_7,954). happensAt(walking(lb2gt_7),954).
orientation(lb2gt_5,152,955). coords(lb2gt_5,92,122,955).  visible(lb2gt_5,955). happensAt(walking(lb2gt_5),955). orientation(lb2gt_7,0,955). coords(lb2gt_7,110,265,955).  visible(lb2gt_7,955). happensAt(walking(lb2gt_7),955).
orientation(lb2gt_5,152,956). coords(lb2gt_5,92,122,956).  visible(lb2gt_5,956). happensAt(walking(lb2gt_5),956). orientation(lb2gt_7,0,956). coords(lb2gt_7,110,265,956).  visible(lb2gt_7,956). happensAt(walking(lb2gt_7),956).
orientation(lb2gt_5,152,957). coords(lb2gt_5,92,122,957).  visible(lb2gt_5,957). happensAt(walking(lb2gt_5),957). orientation(lb2gt_7,0,957). coords(lb2gt_7,110,265,957).  visible(lb2gt_7,957). happensAt(walking(lb2gt_7),957).
orientation(lb2gt_5,152,958). coords(lb2gt_5,93,122,958).  visible(lb2gt_5,958). happensAt(walking(lb2gt_5),958). orientation(lb2gt_7,0,958). coords(lb2gt_7,110,265,958).  visible(lb2gt_7,958). happensAt(walking(lb2gt_7),958).
orientation(lb2gt_5,152,959). coords(lb2gt_5,94,122,959).  visible(lb2gt_5,959). happensAt(walking(lb2gt_5),959). orientation(lb2gt_7,0,959). coords(lb2gt_7,109,264,959).  visible(lb2gt_7,959). happensAt(walking(lb2gt_7),959).
orientation(lb2gt_5,152,960). coords(lb2gt_5,97,122,960).  visible(lb2gt_5,960). happensAt(walking(lb2gt_5),960). orientation(lb2gt_7,0,960). coords(lb2gt_7,109,264,960).  visible(lb2gt_7,960). happensAt(walking(lb2gt_7),960).
orientation(lb2gt_5,152,961). coords(lb2gt_5,98,123,961).  visible(lb2gt_5,961). happensAt(walking(lb2gt_5),961). orientation(lb2gt_7,0,961). coords(lb2gt_7,109,264,961).  visible(lb2gt_7,961). happensAt(walking(lb2gt_7),961).
orientation(lb2gt_5,152,962). coords(lb2gt_5,98,123,962).  visible(lb2gt_5,962). happensAt(walking(lb2gt_5),962). orientation(lb2gt_7,0,962). coords(lb2gt_7,109,263,962).  visible(lb2gt_7,962). happensAt(walking(lb2gt_7),962).
orientation(lb2gt_5,152,963). coords(lb2gt_5,99,124,963).  visible(lb2gt_5,963). happensAt(walking(lb2gt_5),963). orientation(lb2gt_7,0,963). coords(lb2gt_7,109,263,963).  visible(lb2gt_7,963). happensAt(walking(lb2gt_7),963).
orientation(lb2gt_5,152,964). coords(lb2gt_5,100,124,964).  visible(lb2gt_5,964). happensAt(walking(lb2gt_5),964). orientation(lb2gt_7,0,964). coords(lb2gt_7,109,263,964).  visible(lb2gt_7,964). happensAt(walking(lb2gt_7),964).
orientation(lb2gt_5,152,965). coords(lb2gt_5,101,125,965).  visible(lb2gt_5,965). happensAt(walking(lb2gt_5),965). orientation(lb2gt_7,0,965). coords(lb2gt_7,109,263,965).  visible(lb2gt_7,965). happensAt(walking(lb2gt_7),965).
orientation(lb2gt_5,152,966). coords(lb2gt_5,101,125,966).  visible(lb2gt_5,966). happensAt(walking(lb2gt_5),966). orientation(lb2gt_7,0,966). coords(lb2gt_7,108,263,966).  visible(lb2gt_7,966). happensAt(walking(lb2gt_7),966).
orientation(lb2gt_5,152,967). coords(lb2gt_5,101,125,967).  visible(lb2gt_5,967). happensAt(walking(lb2gt_5),967). orientation(lb2gt_7,0,967). coords(lb2gt_7,108,262,967).  visible(lb2gt_7,967). happensAt(walking(lb2gt_7),967).
orientation(lb2gt_5,152,968). coords(lb2gt_5,101,125,968).  visible(lb2gt_5,968). happensAt(walking(lb2gt_5),968). orientation(lb2gt_7,0,968). coords(lb2gt_7,107,262,968).  visible(lb2gt_7,968). happensAt(walking(lb2gt_7),968).
orientation(lb2gt_5,152,969). coords(lb2gt_5,101,127,969).  visible(lb2gt_5,969). happensAt(walking(lb2gt_5),969). orientation(lb2gt_7,0,969). coords(lb2gt_7,107,262,969).  visible(lb2gt_7,969). happensAt(walking(lb2gt_7),969).
orientation(lb2gt_5,152,970). coords(lb2gt_5,102,127,970).  visible(lb2gt_5,970). happensAt(walking(lb2gt_5),970). orientation(lb2gt_7,0,970). coords(lb2gt_7,106,262,970).  visible(lb2gt_7,970). happensAt(walking(lb2gt_7),970).
#show holdsAt/2.