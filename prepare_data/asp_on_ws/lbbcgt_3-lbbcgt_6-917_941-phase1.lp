
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

	orientation(lbbcgt_3,163,917). coords(lbbcgt_3,37,162,917).  visible(lbbcgt_3,917). happensAt(walking(lbbcgt_3),917). orientation(lbbcgt_6,144,917). coords(lbbcgt_6,109,79,917).  visible(lbbcgt_6,917). happensAt(walking(lbbcgt_6),917).
orientation(lbbcgt_3,163,918). coords(lbbcgt_3,37,164,918).  visible(lbbcgt_3,918). happensAt(walking(lbbcgt_3),918). orientation(lbbcgt_6,144,918). coords(lbbcgt_6,108,79,918).  visible(lbbcgt_6,918). happensAt(walking(lbbcgt_6),918).
orientation(lbbcgt_3,163,919). coords(lbbcgt_3,37,166,919).  visible(lbbcgt_3,919). happensAt(walking(lbbcgt_3),919). orientation(lbbcgt_6,144,919). coords(lbbcgt_6,107,80,919).  visible(lbbcgt_6,919). happensAt(walking(lbbcgt_6),919).
orientation(lbbcgt_3,163,920). coords(lbbcgt_3,37,167,920).  visible(lbbcgt_3,920). happensAt(walking(lbbcgt_3),920). orientation(lbbcgt_6,144,920). coords(lbbcgt_6,107,81,920).  visible(lbbcgt_6,920). happensAt(walking(lbbcgt_6),920).
orientation(lbbcgt_3,163,921). coords(lbbcgt_3,37,168,921).  visible(lbbcgt_3,921). happensAt(walking(lbbcgt_3),921). orientation(lbbcgt_6,144,921). coords(lbbcgt_6,107,81,921).  visible(lbbcgt_6,921). happensAt(walking(lbbcgt_6),921).
orientation(lbbcgt_3,163,922). coords(lbbcgt_3,37,169,922).  visible(lbbcgt_3,922). happensAt(walking(lbbcgt_3),922). orientation(lbbcgt_6,144,922). coords(lbbcgt_6,106,81,922).  visible(lbbcgt_6,922). happensAt(walking(lbbcgt_6),922).
orientation(lbbcgt_3,163,923). coords(lbbcgt_3,37,169,923).  visible(lbbcgt_3,923). happensAt(walking(lbbcgt_3),923). orientation(lbbcgt_6,144,923). coords(lbbcgt_6,105,81,923).  visible(lbbcgt_6,923). happensAt(walking(lbbcgt_6),923).
orientation(lbbcgt_3,163,924). coords(lbbcgt_3,37,170,924).  visible(lbbcgt_3,924). happensAt(walking(lbbcgt_3),924). orientation(lbbcgt_6,144,924). coords(lbbcgt_6,105,81,924).  visible(lbbcgt_6,924). happensAt(walking(lbbcgt_6),924).
orientation(lbbcgt_3,163,925). coords(lbbcgt_3,37,170,925).  visible(lbbcgt_3,925). happensAt(walking(lbbcgt_3),925). orientation(lbbcgt_6,144,925). coords(lbbcgt_6,104,82,925).  visible(lbbcgt_6,925). happensAt(walking(lbbcgt_6),925).
orientation(lbbcgt_3,163,926). coords(lbbcgt_3,36,170,926).  visible(lbbcgt_3,926). happensAt(walking(lbbcgt_3),926). orientation(lbbcgt_6,144,926). coords(lbbcgt_6,103,82,926).  visible(lbbcgt_6,926). happensAt(walking(lbbcgt_6),926).
orientation(lbbcgt_3,163,927). coords(lbbcgt_3,36,170,927).  visible(lbbcgt_3,927). happensAt(walking(lbbcgt_3),927). orientation(lbbcgt_6,144,927). coords(lbbcgt_6,101,82,927).  visible(lbbcgt_6,927). happensAt(walking(lbbcgt_6),927).
orientation(lbbcgt_3,163,928). coords(lbbcgt_3,36,172,928).  visible(lbbcgt_3,928). happensAt(walking(lbbcgt_3),928). orientation(lbbcgt_6,144,928). coords(lbbcgt_6,100,82,928).  visible(lbbcgt_6,928). happensAt(walking(lbbcgt_6),928).
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
#show holdsAt/2.