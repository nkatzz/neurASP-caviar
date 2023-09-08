
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

	orientation(wk2gt_1,133,920). coords(wk2gt_1,118,51,920).  visible(wk2gt_1,920). happensAt(inactive(wk2gt_1),920). orientation(wk2gt_5,156,920). coords(wk2gt_5,66,97,920).  visible(wk2gt_5,920). happensAt(walking(wk2gt_5),920).
orientation(wk2gt_1,133,921). coords(wk2gt_1,118,51,921).  visible(wk2gt_1,921). happensAt(walking(wk2gt_1),921). orientation(wk2gt_5,156,921). coords(wk2gt_5,66,97,921).  visible(wk2gt_5,921). happensAt(walking(wk2gt_5),921).
orientation(wk2gt_1,133,922). coords(wk2gt_1,119,50,922).  visible(wk2gt_1,922). happensAt(walking(wk2gt_1),922). orientation(wk2gt_5,156,922). coords(wk2gt_5,66,97,922).  visible(wk2gt_5,922). happensAt(walking(wk2gt_5),922).
orientation(wk2gt_1,133,923). coords(wk2gt_1,119,50,923).  visible(wk2gt_1,923). happensAt(walking(wk2gt_1),923). orientation(wk2gt_5,156,923). coords(wk2gt_5,66,97,923).  visible(wk2gt_5,923). happensAt(walking(wk2gt_5),923).
orientation(wk2gt_1,133,924). coords(wk2gt_1,120,50,924).  visible(wk2gt_1,924). happensAt(walking(wk2gt_1),924). orientation(wk2gt_5,156,924). coords(wk2gt_5,66,97,924).  visible(wk2gt_5,924). happensAt(walking(wk2gt_5),924).
orientation(wk2gt_1,133,925). coords(wk2gt_1,120,51,925).  visible(wk2gt_1,925). happensAt(walking(wk2gt_1),925). orientation(wk2gt_5,156,925). coords(wk2gt_5,66,97,925).  visible(wk2gt_5,925). happensAt(walking(wk2gt_5),925).
orientation(wk2gt_1,133,926). coords(wk2gt_1,120,51,926).  visible(wk2gt_1,926). happensAt(walking(wk2gt_1),926). orientation(wk2gt_5,156,926). coords(wk2gt_5,67,97,926).  visible(wk2gt_5,926). happensAt(walking(wk2gt_5),926).
orientation(wk2gt_1,133,927). coords(wk2gt_1,120,51,927).  visible(wk2gt_1,927). happensAt(walking(wk2gt_1),927). orientation(wk2gt_5,156,927). coords(wk2gt_5,69,96,927).  visible(wk2gt_5,927). happensAt(walking(wk2gt_5),927).
orientation(wk2gt_1,133,928). coords(wk2gt_1,120,51,928).  visible(wk2gt_1,928). happensAt(walking(wk2gt_1),928). orientation(wk2gt_5,156,928). coords(wk2gt_5,70,97,928).  visible(wk2gt_5,928). happensAt(walking(wk2gt_5),928).
orientation(wk2gt_1,133,929). coords(wk2gt_1,120,51,929).  visible(wk2gt_1,929). happensAt(walking(wk2gt_1),929). orientation(wk2gt_5,156,929). coords(wk2gt_5,70,97,929).  visible(wk2gt_5,929). happensAt(walking(wk2gt_5),929).
orientation(wk2gt_1,133,930). coords(wk2gt_1,120,50,930).  visible(wk2gt_1,930). happensAt(walking(wk2gt_1),930). orientation(wk2gt_5,156,930). coords(wk2gt_5,70,97,930).  visible(wk2gt_5,930). happensAt(walking(wk2gt_5),930).
orientation(wk2gt_1,133,931). coords(wk2gt_1,120,49,931).  visible(wk2gt_1,931). happensAt(walking(wk2gt_1),931). orientation(wk2gt_5,156,931). coords(wk2gt_5,70,97,931).  visible(wk2gt_5,931). happensAt(walking(wk2gt_5),931).
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
#show holdsAt/2.