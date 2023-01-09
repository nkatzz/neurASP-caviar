
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

	orientation(br1gt_2,171,946). coords(br1gt_2,91,200,946).  visible(br1gt_2,946). happensAt(walking(br1gt_2),946). orientation(br1gt_3,89,946). coords(br1gt_3,306,94,946).  visible(br1gt_3,946). happensAt(active(br1gt_3),946).
orientation(br1gt_2,171,947). coords(br1gt_2,91,201,947).  visible(br1gt_2,947). happensAt(walking(br1gt_2),947). orientation(br1gt_3,89,947). coords(br1gt_3,306,94,947).  visible(br1gt_3,947). happensAt(active(br1gt_3),947).
orientation(br1gt_2,171,948). coords(br1gt_2,92,201,948).  visible(br1gt_2,948). happensAt(walking(br1gt_2),948). orientation(br1gt_3,89,948). coords(br1gt_3,306,93,948).  visible(br1gt_3,948). happensAt(active(br1gt_3),948).
orientation(br1gt_2,171,949). coords(br1gt_2,92,201,949).  visible(br1gt_2,949). happensAt(walking(br1gt_2),949). orientation(br1gt_3,89,949). coords(br1gt_3,306,93,949).  visible(br1gt_3,949). happensAt(active(br1gt_3),949).
orientation(br1gt_2,171,950). coords(br1gt_2,92,201,950).  visible(br1gt_2,950). happensAt(walking(br1gt_2),950). orientation(br1gt_3,89,950). coords(br1gt_3,306,93,950).  visible(br1gt_3,950). happensAt(active(br1gt_3),950).
orientation(br1gt_2,171,951). coords(br1gt_2,92,201,951).  visible(br1gt_2,951). happensAt(walking(br1gt_2),951). orientation(br1gt_3,89,951). coords(br1gt_3,306,93,951).  visible(br1gt_3,951). happensAt(active(br1gt_3),951).
orientation(br1gt_2,171,952). coords(br1gt_2,93,201,952).  visible(br1gt_2,952). happensAt(walking(br1gt_2),952). orientation(br1gt_3,89,952). coords(br1gt_3,306,93,952).  visible(br1gt_3,952). happensAt(active(br1gt_3),952).
orientation(br1gt_2,171,953). coords(br1gt_2,93,201,953).  visible(br1gt_2,953). happensAt(walking(br1gt_2),953). orientation(br1gt_3,89,953). coords(br1gt_3,306,93,953).  visible(br1gt_3,953). happensAt(active(br1gt_3),953).
orientation(br1gt_2,171,954). coords(br1gt_2,93,201,954).  visible(br1gt_2,954). happensAt(walking(br1gt_2),954). orientation(br1gt_3,89,954). coords(br1gt_3,306,93,954).  visible(br1gt_3,954). happensAt(active(br1gt_3),954).
orientation(br1gt_2,171,955). coords(br1gt_2,93,201,955).  visible(br1gt_2,955). happensAt(walking(br1gt_2),955). orientation(br1gt_3,89,955). coords(br1gt_3,306,93,955).  visible(br1gt_3,955). happensAt(active(br1gt_3),955).
orientation(br1gt_2,171,956). coords(br1gt_2,93,201,956).  visible(br1gt_2,956). happensAt(walking(br1gt_2),956). orientation(br1gt_3,89,956). coords(br1gt_3,306,93,956).  visible(br1gt_3,956). happensAt(active(br1gt_3),956).
orientation(br1gt_2,171,957). coords(br1gt_2,93,201,957).  visible(br1gt_2,957). happensAt(walking(br1gt_2),957). orientation(br1gt_3,89,957). coords(br1gt_3,306,93,957).  visible(br1gt_3,957). happensAt(active(br1gt_3),957).
orientation(br1gt_2,171,958). coords(br1gt_2,93,202,958).  visible(br1gt_2,958). happensAt(walking(br1gt_2),958). orientation(br1gt_3,89,958). coords(br1gt_3,306,93,958).  visible(br1gt_3,958). happensAt(active(br1gt_3),958).
orientation(br1gt_2,171,959). coords(br1gt_2,93,202,959).  visible(br1gt_2,959). happensAt(walking(br1gt_2),959). orientation(br1gt_3,89,959). coords(br1gt_3,306,93,959).  visible(br1gt_3,959). happensAt(active(br1gt_3),959).
orientation(br1gt_2,171,960). coords(br1gt_2,94,202,960).  visible(br1gt_2,960). happensAt(walking(br1gt_2),960). orientation(br1gt_3,89,960). coords(br1gt_3,306,93,960).  visible(br1gt_3,960). happensAt(active(br1gt_3),960).
orientation(br1gt_2,171,961). coords(br1gt_2,94,202,961).  visible(br1gt_2,961). happensAt(walking(br1gt_2),961). orientation(br1gt_3,89,961). coords(br1gt_3,306,93,961).  visible(br1gt_3,961). happensAt(active(br1gt_3),961).
orientation(br1gt_2,171,962). coords(br1gt_2,96,204,962).  visible(br1gt_2,962). happensAt(walking(br1gt_2),962). orientation(br1gt_3,89,962). coords(br1gt_3,306,93,962).  visible(br1gt_3,962). happensAt(active(br1gt_3),962).
orientation(br1gt_2,171,963). coords(br1gt_2,96,204,963).  visible(br1gt_2,963). happensAt(walking(br1gt_2),963). orientation(br1gt_3,89,963). coords(br1gt_3,305,93,963).  visible(br1gt_3,963). happensAt(active(br1gt_3),963).
orientation(br1gt_2,166,964). coords(br1gt_2,96,204,964).  visible(br1gt_2,964). happensAt(walking(br1gt_2),964). orientation(br1gt_3,89,964). coords(br1gt_3,304,93,964).  visible(br1gt_3,964). happensAt(active(br1gt_3),964).
orientation(br1gt_2,166,965). coords(br1gt_2,98,205,965).  visible(br1gt_2,965). happensAt(walking(br1gt_2),965). orientation(br1gt_3,89,965). coords(br1gt_3,304,93,965).  visible(br1gt_3,965). happensAt(active(br1gt_3),965).
orientation(br1gt_2,166,966). coords(br1gt_2,98,205,966).  visible(br1gt_2,966). happensAt(walking(br1gt_2),966). orientation(br1gt_3,89,966). coords(br1gt_3,304,92,966).  visible(br1gt_3,966). happensAt(active(br1gt_3),966).
orientation(br1gt_2,166,967). coords(br1gt_2,99,205,967).  visible(br1gt_2,967). happensAt(walking(br1gt_2),967). orientation(br1gt_3,89,967). coords(br1gt_3,303,92,967).  visible(br1gt_3,967). happensAt(active(br1gt_3),967).
orientation(br1gt_2,166,968). coords(br1gt_2,99,206,968).  visible(br1gt_2,968). happensAt(walking(br1gt_2),968). orientation(br1gt_3,89,968). coords(br1gt_3,302,91,968).  visible(br1gt_3,968). happensAt(walking(br1gt_3),968).
orientation(br1gt_2,166,969). coords(br1gt_2,99,206,969).  visible(br1gt_2,969). happensAt(walking(br1gt_2),969). orientation(br1gt_3,89,969). coords(br1gt_3,302,91,969).  visible(br1gt_3,969). happensAt(walking(br1gt_3),969).
orientation(br1gt_2,166,970). coords(br1gt_2,99,206,970).  visible(br1gt_2,970). happensAt(walking(br1gt_2),970). orientation(br1gt_3,89,970). coords(br1gt_3,302,91,970).  visible(br1gt_3,970). happensAt(walking(br1gt_3),970).
#show holdsAt/2.