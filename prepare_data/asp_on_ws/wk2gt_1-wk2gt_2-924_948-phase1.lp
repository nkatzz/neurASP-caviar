
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

	orientation(wk2gt_1,133,924). coords(wk2gt_1,120,50,924).  visible(wk2gt_1,924). happensAt(walking(wk2gt_1),924). orientation(wk2gt_2,142,924). coords(wk2gt_2,99,63,924).  visible(wk2gt_2,924). happensAt(inactive(wk2gt_2),924).
orientation(wk2gt_1,133,925). coords(wk2gt_1,120,51,925).  visible(wk2gt_1,925). happensAt(walking(wk2gt_1),925). orientation(wk2gt_2,142,925). coords(wk2gt_2,99,63,925).  visible(wk2gt_2,925). happensAt(inactive(wk2gt_2),925).
orientation(wk2gt_1,133,926). coords(wk2gt_1,120,51,926).  visible(wk2gt_1,926). happensAt(walking(wk2gt_1),926). orientation(wk2gt_2,142,926). coords(wk2gt_2,98,63,926).  visible(wk2gt_2,926). happensAt(inactive(wk2gt_2),926).
orientation(wk2gt_1,133,927). coords(wk2gt_1,120,51,927).  visible(wk2gt_1,927). happensAt(walking(wk2gt_1),927). orientation(wk2gt_2,142,927). coords(wk2gt_2,98,63,927).  visible(wk2gt_2,927). happensAt(inactive(wk2gt_2),927).
orientation(wk2gt_1,133,928). coords(wk2gt_1,120,51,928).  visible(wk2gt_1,928). happensAt(walking(wk2gt_1),928). orientation(wk2gt_2,142,928). coords(wk2gt_2,98,63,928).  visible(wk2gt_2,928). happensAt(inactive(wk2gt_2),928).
orientation(wk2gt_1,133,929). coords(wk2gt_1,120,51,929).  visible(wk2gt_1,929). happensAt(walking(wk2gt_1),929). orientation(wk2gt_2,142,929). coords(wk2gt_2,98,63,929).  visible(wk2gt_2,929). happensAt(inactive(wk2gt_2),929).
orientation(wk2gt_1,133,930). coords(wk2gt_1,120,50,930).  visible(wk2gt_1,930). happensAt(walking(wk2gt_1),930). orientation(wk2gt_2,142,930). coords(wk2gt_2,98,63,930).  visible(wk2gt_2,930). happensAt(inactive(wk2gt_2),930).
orientation(wk2gt_1,133,931). coords(wk2gt_1,120,49,931).  visible(wk2gt_1,931). happensAt(walking(wk2gt_1),931). orientation(wk2gt_2,142,931). coords(wk2gt_2,98,63,931).  visible(wk2gt_2,931). happensAt(inactive(wk2gt_2),931).
orientation(wk2gt_1,133,932). coords(wk2gt_1,120,49,932).  visible(wk2gt_1,932). happensAt(walking(wk2gt_1),932). orientation(wk2gt_2,142,932). coords(wk2gt_2,98,63,932).  visible(wk2gt_2,932). happensAt(inactive(wk2gt_2),932).
orientation(wk2gt_1,133,933). coords(wk2gt_1,120,49,933).  visible(wk2gt_1,933). happensAt(walking(wk2gt_1),933). orientation(wk2gt_2,142,933). coords(wk2gt_2,98,63,933).  visible(wk2gt_2,933). happensAt(inactive(wk2gt_2),933).
orientation(wk2gt_1,133,934). coords(wk2gt_1,120,48,934).  visible(wk2gt_1,934). happensAt(walking(wk2gt_1),934). orientation(wk2gt_2,142,934). coords(wk2gt_2,98,64,934).  visible(wk2gt_2,934). happensAt(inactive(wk2gt_2),934).
orientation(wk2gt_1,133,935). coords(wk2gt_1,120,48,935).  visible(wk2gt_1,935). happensAt(walking(wk2gt_1),935). orientation(wk2gt_2,142,935). coords(wk2gt_2,98,64,935).  visible(wk2gt_2,935). happensAt(inactive(wk2gt_2),935).
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
#show holdsAt/2.