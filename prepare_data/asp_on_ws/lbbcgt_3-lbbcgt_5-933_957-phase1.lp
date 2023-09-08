
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

	orientation(lbbcgt_3,163,933). coords(lbbcgt_3,35,176,933).  visible(lbbcgt_3,933). happensAt(walking(lbbcgt_3),933). orientation(lbbcgt_5,158,933). coords(lbbcgt_5,203,187,933).  visible(lbbcgt_5,933). happensAt(walking(lbbcgt_5),933).
orientation(lbbcgt_3,163,934). coords(lbbcgt_3,35,178,934).  visible(lbbcgt_3,934). happensAt(walking(lbbcgt_3),934). orientation(lbbcgt_5,158,934). coords(lbbcgt_5,203,186,934).  visible(lbbcgt_5,934). happensAt(walking(lbbcgt_5),934).
orientation(lbbcgt_3,163,935). coords(lbbcgt_3,35,178,935).  visible(lbbcgt_3,935). happensAt(walking(lbbcgt_3),935). orientation(lbbcgt_5,158,935). coords(lbbcgt_5,203,185,935).  visible(lbbcgt_5,935). happensAt(walking(lbbcgt_5),935).
orientation(lbbcgt_3,163,936). coords(lbbcgt_3,35,179,936).  visible(lbbcgt_3,936). happensAt(walking(lbbcgt_3),936). orientation(lbbcgt_5,158,936). coords(lbbcgt_5,202,184,936).  visible(lbbcgt_5,936). happensAt(walking(lbbcgt_5),936).
orientation(lbbcgt_3,163,937). coords(lbbcgt_3,35,179,937).  visible(lbbcgt_3,937). happensAt(walking(lbbcgt_3),937). orientation(lbbcgt_5,158,937). coords(lbbcgt_5,202,183,937).  visible(lbbcgt_5,937). happensAt(walking(lbbcgt_5),937).
orientation(lbbcgt_3,163,938). coords(lbbcgt_3,35,179,938).  visible(lbbcgt_3,938). happensAt(walking(lbbcgt_3),938). orientation(lbbcgt_5,158,938). coords(lbbcgt_5,201,182,938).  visible(lbbcgt_5,938). happensAt(walking(lbbcgt_5),938).
orientation(lbbcgt_3,163,939). coords(lbbcgt_3,36,179,939).  visible(lbbcgt_3,939). happensAt(walking(lbbcgt_3),939). orientation(lbbcgt_5,158,939). coords(lbbcgt_5,201,180,939).  visible(lbbcgt_5,939). happensAt(walking(lbbcgt_5),939).
orientation(lbbcgt_3,163,940). coords(lbbcgt_3,36,179,940).  visible(lbbcgt_3,940). happensAt(walking(lbbcgt_3),940). orientation(lbbcgt_5,158,940). coords(lbbcgt_5,199,177,940).  visible(lbbcgt_5,940). happensAt(walking(lbbcgt_5),940).
orientation(lbbcgt_3,163,941). coords(lbbcgt_3,35,180,941).  visible(lbbcgt_3,941). happensAt(walking(lbbcgt_3),941). orientation(lbbcgt_5,158,941). coords(lbbcgt_5,198,175,941).  visible(lbbcgt_5,941). happensAt(walking(lbbcgt_5),941).
orientation(lbbcgt_3,163,942). coords(lbbcgt_3,35,180,942).  visible(lbbcgt_3,942). happensAt(walking(lbbcgt_3),942). orientation(lbbcgt_5,158,942). coords(lbbcgt_5,197,173,942).  visible(lbbcgt_5,942). happensAt(walking(lbbcgt_5),942).
orientation(lbbcgt_3,163,943). coords(lbbcgt_3,35,181,943).  visible(lbbcgt_3,943). happensAt(walking(lbbcgt_3),943). orientation(lbbcgt_5,158,943). coords(lbbcgt_5,195,172,943).  visible(lbbcgt_5,943). happensAt(walking(lbbcgt_5),943).
orientation(lbbcgt_3,163,944). coords(lbbcgt_3,35,182,944).  visible(lbbcgt_3,944). happensAt(walking(lbbcgt_3),944). orientation(lbbcgt_5,158,944). coords(lbbcgt_5,193,171,944).  visible(lbbcgt_5,944). happensAt(walking(lbbcgt_5),944).
orientation(lbbcgt_3,163,945). coords(lbbcgt_3,35,184,945).  visible(lbbcgt_3,945). happensAt(walking(lbbcgt_3),945). orientation(lbbcgt_5,158,945). coords(lbbcgt_5,192,169,945).  visible(lbbcgt_5,945). happensAt(walking(lbbcgt_5),945).
orientation(lbbcgt_3,163,946). coords(lbbcgt_3,36,185,946).  visible(lbbcgt_3,946). happensAt(walking(lbbcgt_3),946). orientation(lbbcgt_5,158,946). coords(lbbcgt_5,191,169,946).  visible(lbbcgt_5,946). happensAt(walking(lbbcgt_5),946).
orientation(lbbcgt_3,163,947). coords(lbbcgt_3,36,186,947).  visible(lbbcgt_3,947). happensAt(walking(lbbcgt_3),947). orientation(lbbcgt_5,158,947). coords(lbbcgt_5,191,168,947).  visible(lbbcgt_5,947). happensAt(walking(lbbcgt_5),947).
orientation(lbbcgt_3,163,948). coords(lbbcgt_3,36,187,948).  visible(lbbcgt_3,948). happensAt(walking(lbbcgt_3),948). orientation(lbbcgt_5,158,948). coords(lbbcgt_5,191,166,948).  visible(lbbcgt_5,948). happensAt(walking(lbbcgt_5),948).
orientation(lbbcgt_3,163,949). coords(lbbcgt_3,36,188,949).  visible(lbbcgt_3,949). happensAt(walking(lbbcgt_3),949). orientation(lbbcgt_5,158,949). coords(lbbcgt_5,191,166,949).  visible(lbbcgt_5,949). happensAt(walking(lbbcgt_5),949).
orientation(lbbcgt_3,163,950). coords(lbbcgt_3,36,188,950).  visible(lbbcgt_3,950). happensAt(walking(lbbcgt_3),950). orientation(lbbcgt_5,158,950). coords(lbbcgt_5,191,163,950).  visible(lbbcgt_5,950). happensAt(walking(lbbcgt_5),950).
orientation(lbbcgt_3,163,951). coords(lbbcgt_3,36,189,951).  visible(lbbcgt_3,951). happensAt(walking(lbbcgt_3),951). orientation(lbbcgt_5,158,951). coords(lbbcgt_5,190,162,951).  visible(lbbcgt_5,951). happensAt(walking(lbbcgt_5),951).
orientation(lbbcgt_3,163,952). coords(lbbcgt_3,36,189,952).  visible(lbbcgt_3,952). happensAt(walking(lbbcgt_3),952). orientation(lbbcgt_5,158,952). coords(lbbcgt_5,190,161,952).  visible(lbbcgt_5,952). happensAt(walking(lbbcgt_5),952).
orientation(lbbcgt_3,163,953). coords(lbbcgt_3,35,190,953).  visible(lbbcgt_3,953). happensAt(walking(lbbcgt_3),953). orientation(lbbcgt_5,158,953). coords(lbbcgt_5,189,160,953).  visible(lbbcgt_5,953). happensAt(walking(lbbcgt_5),953).
orientation(lbbcgt_3,163,954). coords(lbbcgt_3,35,190,954).  visible(lbbcgt_3,954). happensAt(walking(lbbcgt_3),954). orientation(lbbcgt_5,158,954). coords(lbbcgt_5,189,160,954).  visible(lbbcgt_5,954). happensAt(walking(lbbcgt_5),954).
orientation(lbbcgt_3,163,955). coords(lbbcgt_3,35,190,955).  visible(lbbcgt_3,955). happensAt(walking(lbbcgt_3),955). orientation(lbbcgt_5,158,955). coords(lbbcgt_5,188,158,955).  visible(lbbcgt_5,955). happensAt(walking(lbbcgt_5),955).
orientation(lbbcgt_3,163,956). coords(lbbcgt_3,35,191,956).  visible(lbbcgt_3,956). happensAt(walking(lbbcgt_3),956). orientation(lbbcgt_5,158,956). coords(lbbcgt_5,188,156,956).  visible(lbbcgt_5,956). happensAt(walking(lbbcgt_5),956).
orientation(lbbcgt_3,163,957). coords(lbbcgt_3,35,191,957).  visible(lbbcgt_3,957). happensAt(walking(lbbcgt_3),957). orientation(lbbcgt_5,158,957). coords(lbbcgt_5,188,155,957).  visible(lbbcgt_5,957). happensAt(walking(lbbcgt_5),957).
#show holdsAt/2.