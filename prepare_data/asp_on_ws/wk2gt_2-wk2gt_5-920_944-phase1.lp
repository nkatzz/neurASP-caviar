
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

	orientation(wk2gt_2,142,920). coords(wk2gt_2,99,63,920).  visible(wk2gt_2,920). happensAt(inactive(wk2gt_2),920). orientation(wk2gt_5,156,920). coords(wk2gt_5,66,97,920).  visible(wk2gt_5,920). happensAt(walking(wk2gt_5),920).
orientation(wk2gt_2,142,921). coords(wk2gt_2,99,63,921).  visible(wk2gt_2,921). happensAt(inactive(wk2gt_2),921). orientation(wk2gt_5,156,921). coords(wk2gt_5,66,97,921).  visible(wk2gt_5,921). happensAt(walking(wk2gt_5),921).
orientation(wk2gt_2,142,922). coords(wk2gt_2,99,63,922).  visible(wk2gt_2,922). happensAt(inactive(wk2gt_2),922). orientation(wk2gt_5,156,922). coords(wk2gt_5,66,97,922).  visible(wk2gt_5,922). happensAt(walking(wk2gt_5),922).
orientation(wk2gt_2,142,923). coords(wk2gt_2,99,63,923).  visible(wk2gt_2,923). happensAt(inactive(wk2gt_2),923). orientation(wk2gt_5,156,923). coords(wk2gt_5,66,97,923).  visible(wk2gt_5,923). happensAt(walking(wk2gt_5),923).
orientation(wk2gt_2,142,924). coords(wk2gt_2,99,63,924).  visible(wk2gt_2,924). happensAt(inactive(wk2gt_2),924). orientation(wk2gt_5,156,924). coords(wk2gt_5,66,97,924).  visible(wk2gt_5,924). happensAt(walking(wk2gt_5),924).
orientation(wk2gt_2,142,925). coords(wk2gt_2,99,63,925).  visible(wk2gt_2,925). happensAt(inactive(wk2gt_2),925). orientation(wk2gt_5,156,925). coords(wk2gt_5,66,97,925).  visible(wk2gt_5,925). happensAt(walking(wk2gt_5),925).
orientation(wk2gt_2,142,926). coords(wk2gt_2,98,63,926).  visible(wk2gt_2,926). happensAt(inactive(wk2gt_2),926). orientation(wk2gt_5,156,926). coords(wk2gt_5,67,97,926).  visible(wk2gt_5,926). happensAt(walking(wk2gt_5),926).
orientation(wk2gt_2,142,927). coords(wk2gt_2,98,63,927).  visible(wk2gt_2,927). happensAt(inactive(wk2gt_2),927). orientation(wk2gt_5,156,927). coords(wk2gt_5,69,96,927).  visible(wk2gt_5,927). happensAt(walking(wk2gt_5),927).
orientation(wk2gt_2,142,928). coords(wk2gt_2,98,63,928).  visible(wk2gt_2,928). happensAt(inactive(wk2gt_2),928). orientation(wk2gt_5,156,928). coords(wk2gt_5,70,97,928).  visible(wk2gt_5,928). happensAt(walking(wk2gt_5),928).
orientation(wk2gt_2,142,929). coords(wk2gt_2,98,63,929).  visible(wk2gt_2,929). happensAt(inactive(wk2gt_2),929). orientation(wk2gt_5,156,929). coords(wk2gt_5,70,97,929).  visible(wk2gt_5,929). happensAt(walking(wk2gt_5),929).
orientation(wk2gt_2,142,930). coords(wk2gt_2,98,63,930).  visible(wk2gt_2,930). happensAt(inactive(wk2gt_2),930). orientation(wk2gt_5,156,930). coords(wk2gt_5,70,97,930).  visible(wk2gt_5,930). happensAt(walking(wk2gt_5),930).
orientation(wk2gt_2,142,931). coords(wk2gt_2,98,63,931).  visible(wk2gt_2,931). happensAt(inactive(wk2gt_2),931). orientation(wk2gt_5,156,931). coords(wk2gt_5,70,97,931).  visible(wk2gt_5,931). happensAt(walking(wk2gt_5),931).
orientation(wk2gt_2,142,932). coords(wk2gt_2,98,63,932).  visible(wk2gt_2,932). happensAt(inactive(wk2gt_2),932). orientation(wk2gt_5,156,932). coords(wk2gt_5,71,97,932).  visible(wk2gt_5,932). happensAt(walking(wk2gt_5),932).
orientation(wk2gt_2,142,933). coords(wk2gt_2,98,63,933).  visible(wk2gt_2,933). happensAt(inactive(wk2gt_2),933). orientation(wk2gt_5,156,933). coords(wk2gt_5,72,96,933).  visible(wk2gt_5,933). happensAt(walking(wk2gt_5),933).
orientation(wk2gt_2,142,934). coords(wk2gt_2,98,64,934).  visible(wk2gt_2,934). happensAt(inactive(wk2gt_2),934). orientation(wk2gt_5,150,934). coords(wk2gt_5,72,96,934).  visible(wk2gt_5,934). happensAt(walking(wk2gt_5),934).
orientation(wk2gt_2,142,935). coords(wk2gt_2,98,64,935).  visible(wk2gt_2,935). happensAt(inactive(wk2gt_2),935). orientation(wk2gt_5,150,935). coords(wk2gt_5,72,96,935).  visible(wk2gt_5,935). happensAt(walking(wk2gt_5),935).
orientation(wk2gt_2,138,936). coords(wk2gt_2,99,65,936).  visible(wk2gt_2,936). happensAt(active(wk2gt_2),936). orientation(wk2gt_5,150,936). coords(wk2gt_5,72,96,936).  visible(wk2gt_5,936). happensAt(walking(wk2gt_5),936).
orientation(wk2gt_2,138,937). coords(wk2gt_2,99,65,937).  visible(wk2gt_2,937). happensAt(active(wk2gt_2),937). orientation(wk2gt_5,150,937). coords(wk2gt_5,72,97,937).  visible(wk2gt_5,937). happensAt(walking(wk2gt_5),937).
orientation(wk2gt_2,130,938). coords(wk2gt_2,99,64,938).  visible(wk2gt_2,938). happensAt(active(wk2gt_2),938). orientation(wk2gt_5,150,938). coords(wk2gt_5,72,97,938).  visible(wk2gt_5,938). happensAt(walking(wk2gt_5),938).
orientation(wk2gt_2,130,939). coords(wk2gt_2,99,64,939).  visible(wk2gt_2,939). happensAt(active(wk2gt_2),939). orientation(wk2gt_5,150,939). coords(wk2gt_5,72,98,939).  visible(wk2gt_5,939). happensAt(walking(wk2gt_5),939).
orientation(wk2gt_2,130,940). coords(wk2gt_2,99,65,940).  visible(wk2gt_2,940). happensAt(active(wk2gt_2),940). orientation(wk2gt_5,150,940). coords(wk2gt_5,73,98,940).  visible(wk2gt_5,940). happensAt(walking(wk2gt_5),940).
orientation(wk2gt_2,124,941). coords(wk2gt_2,99,65,941).  visible(wk2gt_2,941). happensAt(active(wk2gt_2),941). orientation(wk2gt_5,150,941). coords(wk2gt_5,74,98,941).  visible(wk2gt_5,941). happensAt(walking(wk2gt_5),941).
orientation(wk2gt_2,118,942). coords(wk2gt_2,99,65,942).  visible(wk2gt_2,942). happensAt(active(wk2gt_2),942). orientation(wk2gt_5,150,942). coords(wk2gt_5,74,98,942).  visible(wk2gt_5,942). happensAt(walking(wk2gt_5),942).
orientation(wk2gt_2,118,943). coords(wk2gt_2,100,65,943).  visible(wk2gt_2,943). happensAt(active(wk2gt_2),943). orientation(wk2gt_5,150,943). coords(wk2gt_5,75,98,943).  visible(wk2gt_5,943). happensAt(walking(wk2gt_5),943).
orientation(wk2gt_2,118,944). coords(wk2gt_2,100,65,944).  visible(wk2gt_2,944). happensAt(active(wk2gt_2),944). orientation(wk2gt_5,150,944). coords(wk2gt_5,76,99,944).  visible(wk2gt_5,944). happensAt(walking(wk2gt_5),944).
#show holdsAt/2.