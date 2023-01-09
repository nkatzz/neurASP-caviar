
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

	orientation(lb1gt_3,145,945). coords(lb1gt_3,294,96,945).  visible(lb1gt_3,945). happensAt(walking(lb1gt_3),945). orientation(lb1gt_4,90,945). coords(lb1gt_4,310,112,945).  happensAt(appear(lb1gt_4),945). happensAt(inactive(lb1gt_4),945).
orientation(lb1gt_3,142,946). coords(lb1gt_3,294,96,946).  visible(lb1gt_3,946). happensAt(walking(lb1gt_3),946). orientation(lb1gt_4,90,946). coords(lb1gt_4,310,112,946).  visible(lb1gt_4,946). happensAt(inactive(lb1gt_4),946).
orientation(lb1gt_3,146,947). coords(lb1gt_3,292,96,947).  visible(lb1gt_3,947). happensAt(walking(lb1gt_3),947). orientation(lb1gt_4,90,947). coords(lb1gt_4,310,112,947).  visible(lb1gt_4,947). happensAt(inactive(lb1gt_4),947).
orientation(lb1gt_3,138,948). coords(lb1gt_3,291,95,948).  visible(lb1gt_3,948). happensAt(walking(lb1gt_3),948). orientation(lb1gt_4,90,948). coords(lb1gt_4,310,112,948).  visible(lb1gt_4,948). happensAt(inactive(lb1gt_4),948).
orientation(lb1gt_3,139,949). coords(lb1gt_3,291,95,949).  visible(lb1gt_3,949). happensAt(walking(lb1gt_3),949). orientation(lb1gt_4,90,949). coords(lb1gt_4,310,112,949).  visible(lb1gt_4,949). happensAt(inactive(lb1gt_4),949).
orientation(lb1gt_3,141,950). coords(lb1gt_3,290,95,950).  visible(lb1gt_3,950). happensAt(walking(lb1gt_3),950). orientation(lb1gt_4,90,950). coords(lb1gt_4,310,112,950).  visible(lb1gt_4,950). happensAt(inactive(lb1gt_4),950).
orientation(lb1gt_3,139,951). coords(lb1gt_3,289,95,951).  visible(lb1gt_3,951). happensAt(walking(lb1gt_3),951). orientation(lb1gt_4,90,951). coords(lb1gt_4,310,112,951).  visible(lb1gt_4,951). happensAt(inactive(lb1gt_4),951).
orientation(lb1gt_3,131,952). coords(lb1gt_3,289,94,952).  visible(lb1gt_3,952). happensAt(walking(lb1gt_3),952). orientation(lb1gt_4,90,952). coords(lb1gt_4,310,112,952).  visible(lb1gt_4,952). happensAt(inactive(lb1gt_4),952).
orientation(lb1gt_3,131,953). coords(lb1gt_3,289,94,953).  visible(lb1gt_3,953). happensAt(walking(lb1gt_3),953). orientation(lb1gt_4,90,953). coords(lb1gt_4,310,112,953).  visible(lb1gt_4,953). happensAt(inactive(lb1gt_4),953).
orientation(lb1gt_3,130,954). coords(lb1gt_3,289,93,954).  visible(lb1gt_3,954). happensAt(walking(lb1gt_3),954). orientation(lb1gt_4,90,954). coords(lb1gt_4,310,112,954).  visible(lb1gt_4,954). happensAt(inactive(lb1gt_4),954).
orientation(lb1gt_3,120,955). coords(lb1gt_3,288,93,955).  visible(lb1gt_3,955). happensAt(walking(lb1gt_3),955). orientation(lb1gt_4,90,955). coords(lb1gt_4,310,112,955).  visible(lb1gt_4,955). happensAt(inactive(lb1gt_4),955).
orientation(lb1gt_3,117,956). coords(lb1gt_3,288,94,956).  visible(lb1gt_3,956). happensAt(walking(lb1gt_3),956). orientation(lb1gt_4,90,956). coords(lb1gt_4,310,112,956).  visible(lb1gt_4,956). happensAt(inactive(lb1gt_4),956).
orientation(lb1gt_3,114,957). coords(lb1gt_3,287,94,957).  visible(lb1gt_3,957). happensAt(walking(lb1gt_3),957). orientation(lb1gt_4,90,957). coords(lb1gt_4,310,112,957).  visible(lb1gt_4,957). happensAt(inactive(lb1gt_4),957).
orientation(lb1gt_3,114,958). coords(lb1gt_3,286,94,958).  visible(lb1gt_3,958). happensAt(walking(lb1gt_3),958). orientation(lb1gt_4,90,958). coords(lb1gt_4,310,112,958).  visible(lb1gt_4,958). happensAt(inactive(lb1gt_4),958).
orientation(lb1gt_3,110,959). coords(lb1gt_3,286,93,959).  visible(lb1gt_3,959). happensAt(walking(lb1gt_3),959). orientation(lb1gt_4,90,959). coords(lb1gt_4,310,112,959).  visible(lb1gt_4,959). happensAt(inactive(lb1gt_4),959).
orientation(lb1gt_3,106,960). coords(lb1gt_3,286,92,960).  visible(lb1gt_3,960). happensAt(walking(lb1gt_3),960). orientation(lb1gt_4,90,960). coords(lb1gt_4,310,112,960).  visible(lb1gt_4,960). happensAt(inactive(lb1gt_4),960).
orientation(lb1gt_3,108,961). coords(lb1gt_3,284,91,961).  visible(lb1gt_3,961). happensAt(walking(lb1gt_3),961). orientation(lb1gt_4,90,961). coords(lb1gt_4,310,112,961).  visible(lb1gt_4,961). happensAt(inactive(lb1gt_4),961).
orientation(lb1gt_3,103,962). coords(lb1gt_3,283,90,962).  visible(lb1gt_3,962). happensAt(walking(lb1gt_3),962). orientation(lb1gt_4,90,962). coords(lb1gt_4,310,112,962).  visible(lb1gt_4,962). happensAt(inactive(lb1gt_4),962).
orientation(lb1gt_3,103,963). coords(lb1gt_3,282,89,963).  visible(lb1gt_3,963). happensAt(walking(lb1gt_3),963). orientation(lb1gt_4,90,963). coords(lb1gt_4,310,112,963).  visible(lb1gt_4,963). happensAt(inactive(lb1gt_4),963).
orientation(lb1gt_3,111,964). coords(lb1gt_3,282,88,964).  visible(lb1gt_3,964). happensAt(walking(lb1gt_3),964). orientation(lb1gt_4,90,964). coords(lb1gt_4,310,112,964).  visible(lb1gt_4,964). happensAt(inactive(lb1gt_4),964).
orientation(lb1gt_3,117,965). coords(lb1gt_3,280,88,965).  visible(lb1gt_3,965). happensAt(walking(lb1gt_3),965). orientation(lb1gt_4,90,965). coords(lb1gt_4,310,112,965).  visible(lb1gt_4,965). happensAt(inactive(lb1gt_4),965).
orientation(lb1gt_3,123,966). coords(lb1gt_3,280,88,966).  visible(lb1gt_3,966). happensAt(walking(lb1gt_3),966). orientation(lb1gt_4,90,966). coords(lb1gt_4,310,112,966).  visible(lb1gt_4,966). happensAt(inactive(lb1gt_4),966).
orientation(lb1gt_3,125,967). coords(lb1gt_3,278,87,967).  visible(lb1gt_3,967). happensAt(walking(lb1gt_3),967). orientation(lb1gt_4,90,967). coords(lb1gt_4,310,112,967).  visible(lb1gt_4,967). happensAt(inactive(lb1gt_4),967).
orientation(lb1gt_3,125,968). coords(lb1gt_3,277,87,968).  visible(lb1gt_3,968). happensAt(walking(lb1gt_3),968). orientation(lb1gt_4,90,968). coords(lb1gt_4,310,112,968).  visible(lb1gt_4,968). happensAt(inactive(lb1gt_4),968).
orientation(lb1gt_3,125,969). coords(lb1gt_3,276,88,969).  visible(lb1gt_3,969). happensAt(walking(lb1gt_3),969). orientation(lb1gt_4,90,969). coords(lb1gt_4,310,112,969).  visible(lb1gt_4,969). happensAt(inactive(lb1gt_4),969).
#show holdsAt/2.