
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

	orientation(wk2gt_1,133,932). coords(wk2gt_1,120,49,932).  visible(wk2gt_1,932). happensAt(walking(wk2gt_1),932). orientation(wk2gt_5,156,932). coords(wk2gt_5,71,97,932).  visible(wk2gt_5,932). happensAt(walking(wk2gt_5),932).
orientation(wk2gt_1,133,933). coords(wk2gt_1,120,49,933).  visible(wk2gt_1,933). happensAt(walking(wk2gt_1),933). orientation(wk2gt_5,156,933). coords(wk2gt_5,72,96,933).  visible(wk2gt_5,933). happensAt(walking(wk2gt_5),933).
orientation(wk2gt_1,133,934). coords(wk2gt_1,120,48,934).  visible(wk2gt_1,934). happensAt(walking(wk2gt_1),934). orientation(wk2gt_5,150,934). coords(wk2gt_5,72,96,934).  visible(wk2gt_5,934). happensAt(walking(wk2gt_5),934).
orientation(wk2gt_1,133,935). coords(wk2gt_1,120,48,935).  visible(wk2gt_1,935). happensAt(walking(wk2gt_1),935). orientation(wk2gt_5,150,935). coords(wk2gt_5,72,96,935).  visible(wk2gt_5,935). happensAt(walking(wk2gt_5),935).
orientation(wk2gt_1,133,936). coords(wk2gt_1,121,48,936).  visible(wk2gt_1,936). happensAt(walking(wk2gt_1),936). orientation(wk2gt_5,150,936). coords(wk2gt_5,72,96,936).  visible(wk2gt_5,936). happensAt(walking(wk2gt_5),936).
orientation(wk2gt_1,133,937). coords(wk2gt_1,121,48,937).  visible(wk2gt_1,937). happensAt(walking(wk2gt_1),937). orientation(wk2gt_5,150,937). coords(wk2gt_5,72,97,937).  visible(wk2gt_5,937). happensAt(walking(wk2gt_5),937).
orientation(wk2gt_1,133,938). coords(wk2gt_1,121,48,938).  visible(wk2gt_1,938). happensAt(walking(wk2gt_1),938). orientation(wk2gt_5,150,938). coords(wk2gt_5,72,97,938).  visible(wk2gt_5,938). happensAt(walking(wk2gt_5),938).
orientation(wk2gt_1,133,939). coords(wk2gt_1,121,48,939).  visible(wk2gt_1,939). happensAt(walking(wk2gt_1),939). orientation(wk2gt_5,150,939). coords(wk2gt_5,72,98,939).  visible(wk2gt_5,939). happensAt(walking(wk2gt_5),939).
orientation(wk2gt_1,133,940). coords(wk2gt_1,121,48,940).  visible(wk2gt_1,940). happensAt(walking(wk2gt_1),940). orientation(wk2gt_5,150,940). coords(wk2gt_5,73,98,940).  visible(wk2gt_5,940). happensAt(walking(wk2gt_5),940).
orientation(wk2gt_1,133,941). coords(wk2gt_1,121,48,941).  visible(wk2gt_1,941). happensAt(walking(wk2gt_1),941). orientation(wk2gt_5,150,941). coords(wk2gt_5,74,98,941).  visible(wk2gt_5,941). happensAt(walking(wk2gt_5),941).
orientation(wk2gt_1,133,942). coords(wk2gt_1,121,48,942).  visible(wk2gt_1,942). happensAt(walking(wk2gt_1),942). orientation(wk2gt_5,150,942). coords(wk2gt_5,74,98,942).  visible(wk2gt_5,942). happensAt(walking(wk2gt_5),942).
orientation(wk2gt_1,133,943). coords(wk2gt_1,121,48,943).  visible(wk2gt_1,943). happensAt(walking(wk2gt_1),943). orientation(wk2gt_5,150,943). coords(wk2gt_5,75,98,943).  visible(wk2gt_5,943). happensAt(walking(wk2gt_5),943).
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
#show holdsAt/2.