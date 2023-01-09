
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

	orientation(lbbcgt_3,163,929). coords(lbbcgt_3,36,172,929).  visible(lbbcgt_3,929). happensAt(walking(lbbcgt_3),929). orientation(lbbcgt_6,144,929). coords(lbbcgt_6,99,82,929).  visible(lbbcgt_6,929). happensAt(walking(lbbcgt_6),929).
orientation(lbbcgt_3,163,930). coords(lbbcgt_3,36,173,930).  visible(lbbcgt_3,930). happensAt(walking(lbbcgt_3),930). orientation(lbbcgt_6,144,930). coords(lbbcgt_6,97,82,930).  visible(lbbcgt_6,930). happensAt(walking(lbbcgt_6),930).
orientation(lbbcgt_3,163,931). coords(lbbcgt_3,35,173,931).  visible(lbbcgt_3,931). happensAt(walking(lbbcgt_3),931). orientation(lbbcgt_6,144,931). coords(lbbcgt_6,97,82,931).  visible(lbbcgt_6,931). happensAt(walking(lbbcgt_6),931).
orientation(lbbcgt_3,163,932). coords(lbbcgt_3,35,176,932).  visible(lbbcgt_3,932). happensAt(walking(lbbcgt_3),932). orientation(lbbcgt_6,144,932). coords(lbbcgt_6,97,83,932).  visible(lbbcgt_6,932). happensAt(walking(lbbcgt_6),932).
orientation(lbbcgt_3,163,933). coords(lbbcgt_3,35,176,933).  visible(lbbcgt_3,933). happensAt(walking(lbbcgt_3),933). orientation(lbbcgt_6,144,933). coords(lbbcgt_6,96,83,933).  visible(lbbcgt_6,933). happensAt(walking(lbbcgt_6),933).
orientation(lbbcgt_3,163,934). coords(lbbcgt_3,35,178,934).  visible(lbbcgt_3,934). happensAt(walking(lbbcgt_3),934). orientation(lbbcgt_6,144,934). coords(lbbcgt_6,96,83,934).  visible(lbbcgt_6,934). happensAt(walking(lbbcgt_6),934).
orientation(lbbcgt_3,163,935). coords(lbbcgt_3,35,178,935).  visible(lbbcgt_3,935). happensAt(walking(lbbcgt_3),935). orientation(lbbcgt_6,150,935). coords(lbbcgt_6,96,83,935).  visible(lbbcgt_6,935). happensAt(walking(lbbcgt_6),935).
orientation(lbbcgt_3,163,936). coords(lbbcgt_3,35,179,936).  visible(lbbcgt_3,936). happensAt(walking(lbbcgt_3),936). orientation(lbbcgt_6,150,936). coords(lbbcgt_6,95,84,936).  visible(lbbcgt_6,936). happensAt(walking(lbbcgt_6),936).
orientation(lbbcgt_3,163,937). coords(lbbcgt_3,35,179,937).  visible(lbbcgt_3,937). happensAt(walking(lbbcgt_3),937). orientation(lbbcgt_6,150,937). coords(lbbcgt_6,95,84,937).  visible(lbbcgt_6,937). happensAt(walking(lbbcgt_6),937).
orientation(lbbcgt_3,163,938). coords(lbbcgt_3,35,179,938).  visible(lbbcgt_3,938). happensAt(walking(lbbcgt_3),938). orientation(lbbcgt_6,150,938). coords(lbbcgt_6,94,84,938).  visible(lbbcgt_6,938). happensAt(walking(lbbcgt_6),938).
orientation(lbbcgt_3,163,939). coords(lbbcgt_3,36,179,939).  visible(lbbcgt_3,939). happensAt(walking(lbbcgt_3),939). orientation(lbbcgt_6,150,939). coords(lbbcgt_6,94,85,939).  visible(lbbcgt_6,939). happensAt(walking(lbbcgt_6),939).
orientation(lbbcgt_3,163,940). coords(lbbcgt_3,36,179,940).  visible(lbbcgt_3,940). happensAt(walking(lbbcgt_3),940). orientation(lbbcgt_6,150,940). coords(lbbcgt_6,92,84,940).  visible(lbbcgt_6,940). happensAt(walking(lbbcgt_6),940).
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
#show holdsAt/2.