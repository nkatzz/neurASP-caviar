
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

	orientation(wk2gt_1,133,936). coords(wk2gt_1,121,48,936).  visible(wk2gt_1,936). happensAt(walking(wk2gt_1),936). orientation(wk2gt_2,138,936). coords(wk2gt_2,99,65,936).  visible(wk2gt_2,936). happensAt(active(wk2gt_2),936).
orientation(wk2gt_1,133,937). coords(wk2gt_1,121,48,937).  visible(wk2gt_1,937). happensAt(walking(wk2gt_1),937). orientation(wk2gt_2,138,937). coords(wk2gt_2,99,65,937).  visible(wk2gt_2,937). happensAt(active(wk2gt_2),937).
orientation(wk2gt_1,133,938). coords(wk2gt_1,121,48,938).  visible(wk2gt_1,938). happensAt(walking(wk2gt_1),938). orientation(wk2gt_2,130,938). coords(wk2gt_2,99,64,938).  visible(wk2gt_2,938). happensAt(active(wk2gt_2),938).
orientation(wk2gt_1,133,939). coords(wk2gt_1,121,48,939).  visible(wk2gt_1,939). happensAt(walking(wk2gt_1),939). orientation(wk2gt_2,130,939). coords(wk2gt_2,99,64,939).  visible(wk2gt_2,939). happensAt(active(wk2gt_2),939).
orientation(wk2gt_1,133,940). coords(wk2gt_1,121,48,940).  visible(wk2gt_1,940). happensAt(walking(wk2gt_1),940). orientation(wk2gt_2,130,940). coords(wk2gt_2,99,65,940).  visible(wk2gt_2,940). happensAt(active(wk2gt_2),940).
orientation(wk2gt_1,133,941). coords(wk2gt_1,121,48,941).  visible(wk2gt_1,941). happensAt(walking(wk2gt_1),941). orientation(wk2gt_2,124,941). coords(wk2gt_2,99,65,941).  visible(wk2gt_2,941). happensAt(active(wk2gt_2),941).
orientation(wk2gt_1,133,942). coords(wk2gt_1,121,48,942).  visible(wk2gt_1,942). happensAt(walking(wk2gt_1),942). orientation(wk2gt_2,118,942). coords(wk2gt_2,99,65,942).  visible(wk2gt_2,942). happensAt(active(wk2gt_2),942).
orientation(wk2gt_1,133,943). coords(wk2gt_1,121,48,943).  visible(wk2gt_1,943). happensAt(walking(wk2gt_1),943). orientation(wk2gt_2,118,943). coords(wk2gt_2,100,65,943).  visible(wk2gt_2,943). happensAt(active(wk2gt_2),943).
orientation(wk2gt_1,133,944). coords(wk2gt_1,121,48,944).  visible(wk2gt_1,944). happensAt(walking(wk2gt_1),944). orientation(wk2gt_2,118,944). coords(wk2gt_2,100,65,944).  visible(wk2gt_2,944). happensAt(active(wk2gt_2),944).
orientation(wk2gt_1,133,945). coords(wk2gt_1,121,48,945).  visible(wk2gt_1,945). happensAt(walking(wk2gt_1),945). orientation(wk2gt_2,118,945). coords(wk2gt_2,100,65,945).  visible(wk2gt_2,945). happensAt(active(wk2gt_2),945).
orientation(wk2gt_1,133,946). coords(wk2gt_1,121,48,946).  visible(wk2gt_1,946). happensAt(walking(wk2gt_1),946). orientation(wk2gt_2,118,946). coords(wk2gt_2,100,65,946).  visible(wk2gt_2,946). happensAt(active(wk2gt_2),946).
orientation(wk2gt_1,133,947). coords(wk2gt_1,121,48,947).  visible(wk2gt_1,947). happensAt(walking(wk2gt_1),947). orientation(wk2gt_2,118,947). coords(wk2gt_2,100,64,947).  visible(wk2gt_2,947). happensAt(active(wk2gt_2),947).
orientation(wk2gt_1,133,948). coords(wk2gt_1,121,48,948).  visible(wk2gt_1,948). happensAt(walking(wk2gt_1),948). orientation(wk2gt_2,118,948). coords(wk2gt_2,100,64,948).  visible(wk2gt_2,948). happensAt(active(wk2gt_2),948).
orientation(wk2gt_1,133,949). coords(wk2gt_1,121,48,949).  visible(wk2gt_1,949). happensAt(walking(wk2gt_1),949). orientation(wk2gt_2,118,949). coords(wk2gt_2,100,64,949).  visible(wk2gt_2,949). happensAt(active(wk2gt_2),949).
orientation(wk2gt_1,133,950). coords(wk2gt_1,121,48,950).  visible(wk2gt_1,950). happensAt(walking(wk2gt_1),950). orientation(wk2gt_2,118,950). coords(wk2gt_2,100,64,950).  visible(wk2gt_2,950). happensAt(active(wk2gt_2),950).
orientation(wk2gt_1,133,951). coords(wk2gt_1,122,47,951).  visible(wk2gt_1,951). happensAt(walking(wk2gt_1),951). orientation(wk2gt_2,118,951). coords(wk2gt_2,100,64,951).  visible(wk2gt_2,951). happensAt(active(wk2gt_2),951).
orientation(wk2gt_1,133,952). coords(wk2gt_1,122,47,952).  visible(wk2gt_1,952). happensAt(walking(wk2gt_1),952). orientation(wk2gt_2,118,952). coords(wk2gt_2,100,64,952).  visible(wk2gt_2,952). happensAt(active(wk2gt_2),952).
orientation(wk2gt_1,133,953). coords(wk2gt_1,123,48,953).  visible(wk2gt_1,953). happensAt(walking(wk2gt_1),953). orientation(wk2gt_2,118,953). coords(wk2gt_2,100,64,953).  visible(wk2gt_2,953). happensAt(active(wk2gt_2),953).
orientation(wk2gt_1,133,954). coords(wk2gt_1,123,48,954).  visible(wk2gt_1,954). happensAt(walking(wk2gt_1),954). orientation(wk2gt_2,118,954). coords(wk2gt_2,99,64,954).  visible(wk2gt_2,954). happensAt(active(wk2gt_2),954).
orientation(wk2gt_1,133,955). coords(wk2gt_1,123,47,955).  visible(wk2gt_1,955). happensAt(walking(wk2gt_1),955). orientation(wk2gt_2,122,955). coords(wk2gt_2,99,64,955).  visible(wk2gt_2,955). happensAt(active(wk2gt_2),955).
orientation(wk2gt_1,133,956). coords(wk2gt_1,123,46,956).  visible(wk2gt_1,956). happensAt(walking(wk2gt_1),956). orientation(wk2gt_2,122,956). coords(wk2gt_2,99,64,956).  visible(wk2gt_2,956). happensAt(active(wk2gt_2),956).
orientation(wk2gt_1,133,957). coords(wk2gt_1,123,46,957).  visible(wk2gt_1,957). happensAt(walking(wk2gt_1),957). orientation(wk2gt_2,126,957). coords(wk2gt_2,99,64,957).  visible(wk2gt_2,957). happensAt(active(wk2gt_2),957).
orientation(wk2gt_1,133,958). coords(wk2gt_1,124,46,958).  visible(wk2gt_1,958). happensAt(walking(wk2gt_1),958). orientation(wk2gt_2,126,958). coords(wk2gt_2,99,64,958).  visible(wk2gt_2,958). happensAt(active(wk2gt_2),958).
orientation(wk2gt_1,133,959). coords(wk2gt_1,124,46,959).  visible(wk2gt_1,959). happensAt(walking(wk2gt_1),959). orientation(wk2gt_2,128,959). coords(wk2gt_2,99,64,959).  visible(wk2gt_2,959). happensAt(active(wk2gt_2),959).
orientation(wk2gt_1,133,960). coords(wk2gt_1,125,45,960).  visible(wk2gt_1,960). happensAt(walking(wk2gt_1),960). orientation(wk2gt_2,128,960). coords(wk2gt_2,99,64,960).  visible(wk2gt_2,960). happensAt(active(wk2gt_2),960).
#show holdsAt/2.