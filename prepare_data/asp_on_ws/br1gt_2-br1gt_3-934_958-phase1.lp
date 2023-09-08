
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

	orientation(br1gt_2,171,934). coords(br1gt_2,86,199,934).  visible(br1gt_2,934). happensAt(walking(br1gt_2),934). orientation(br1gt_3,89,934). coords(br1gt_3,301,94,934).  visible(br1gt_3,934). happensAt(active(br1gt_3),934).
orientation(br1gt_2,171,935). coords(br1gt_2,86,199,935).  visible(br1gt_2,935). happensAt(walking(br1gt_2),935). orientation(br1gt_3,89,935). coords(br1gt_3,303,95,935).  visible(br1gt_3,935). happensAt(active(br1gt_3),935).
orientation(br1gt_2,171,936). coords(br1gt_2,86,199,936).  visible(br1gt_2,936). happensAt(walking(br1gt_2),936). orientation(br1gt_3,89,936). coords(br1gt_3,303,94,936).  visible(br1gt_3,936). happensAt(active(br1gt_3),936).
orientation(br1gt_2,171,937). coords(br1gt_2,86,199,937).  visible(br1gt_2,937). happensAt(walking(br1gt_2),937). orientation(br1gt_3,89,937). coords(br1gt_3,304,94,937).  visible(br1gt_3,937). happensAt(active(br1gt_3),937).
orientation(br1gt_2,171,938). coords(br1gt_2,86,199,938).  visible(br1gt_2,938). happensAt(walking(br1gt_2),938). orientation(br1gt_3,89,938). coords(br1gt_3,304,94,938).  visible(br1gt_3,938). happensAt(active(br1gt_3),938).
orientation(br1gt_2,171,939). coords(br1gt_2,87,199,939).  visible(br1gt_2,939). happensAt(walking(br1gt_2),939). orientation(br1gt_3,89,939). coords(br1gt_3,304,94,939).  visible(br1gt_3,939). happensAt(active(br1gt_3),939).
orientation(br1gt_2,171,940). coords(br1gt_2,88,199,940).  visible(br1gt_2,940). happensAt(walking(br1gt_2),940). orientation(br1gt_3,89,940). coords(br1gt_3,304,94,940).  visible(br1gt_3,940). happensAt(active(br1gt_3),940).
orientation(br1gt_2,171,941). coords(br1gt_2,88,200,941).  visible(br1gt_2,941). happensAt(walking(br1gt_2),941). orientation(br1gt_3,89,941). coords(br1gt_3,304,94,941).  visible(br1gt_3,941). happensAt(active(br1gt_3),941).
orientation(br1gt_2,171,942). coords(br1gt_2,89,200,942).  visible(br1gt_2,942). happensAt(walking(br1gt_2),942). orientation(br1gt_3,89,942). coords(br1gt_3,304,94,942).  visible(br1gt_3,942). happensAt(active(br1gt_3),942).
orientation(br1gt_2,171,943). coords(br1gt_2,90,201,943).  visible(br1gt_2,943). happensAt(walking(br1gt_2),943). orientation(br1gt_3,89,943). coords(br1gt_3,306,94,943).  visible(br1gt_3,943). happensAt(active(br1gt_3),943).
orientation(br1gt_2,171,944). coords(br1gt_2,90,200,944).  visible(br1gt_2,944). happensAt(walking(br1gt_2),944). orientation(br1gt_3,89,944). coords(br1gt_3,306,94,944).  visible(br1gt_3,944). happensAt(active(br1gt_3),944).
orientation(br1gt_2,171,945). coords(br1gt_2,90,200,945).  visible(br1gt_2,945). happensAt(walking(br1gt_2),945). orientation(br1gt_3,89,945). coords(br1gt_3,306,94,945).  visible(br1gt_3,945). happensAt(active(br1gt_3),945).
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
#show holdsAt/2.