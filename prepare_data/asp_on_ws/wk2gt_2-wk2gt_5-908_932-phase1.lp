
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

	orientation(wk2gt_2,142,908). coords(wk2gt_2,99,63,908).  visible(wk2gt_2,908). happensAt(inactive(wk2gt_2),908). orientation(wk2gt_5,156,908). coords(wk2gt_5,61,96,908).  visible(wk2gt_5,908). happensAt(walking(wk2gt_5),908).
orientation(wk2gt_2,142,909). coords(wk2gt_2,99,63,909).  visible(wk2gt_2,909). happensAt(inactive(wk2gt_2),909). orientation(wk2gt_5,156,909). coords(wk2gt_5,62,96,909).  visible(wk2gt_5,909). happensAt(walking(wk2gt_5),909).
orientation(wk2gt_2,142,910). coords(wk2gt_2,99,63,910).  visible(wk2gt_2,910). happensAt(inactive(wk2gt_2),910). orientation(wk2gt_5,156,910). coords(wk2gt_5,63,96,910).  visible(wk2gt_5,910). happensAt(walking(wk2gt_5),910).
orientation(wk2gt_2,142,911). coords(wk2gt_2,99,63,911).  visible(wk2gt_2,911). happensAt(inactive(wk2gt_2),911). orientation(wk2gt_5,156,911). coords(wk2gt_5,63,96,911).  visible(wk2gt_5,911). happensAt(walking(wk2gt_5),911).
orientation(wk2gt_2,142,912). coords(wk2gt_2,99,63,912).  visible(wk2gt_2,912). happensAt(inactive(wk2gt_2),912). orientation(wk2gt_5,156,912). coords(wk2gt_5,63,96,912).  visible(wk2gt_5,912). happensAt(walking(wk2gt_5),912).
orientation(wk2gt_2,142,913). coords(wk2gt_2,99,63,913).  visible(wk2gt_2,913). happensAt(inactive(wk2gt_2),913). orientation(wk2gt_5,156,913). coords(wk2gt_5,63,96,913).  visible(wk2gt_5,913). happensAt(walking(wk2gt_5),913).
orientation(wk2gt_2,142,914). coords(wk2gt_2,99,63,914).  visible(wk2gt_2,914). happensAt(inactive(wk2gt_2),914). orientation(wk2gt_5,156,914). coords(wk2gt_5,63,96,914).  visible(wk2gt_5,914). happensAt(walking(wk2gt_5),914).
orientation(wk2gt_2,142,915). coords(wk2gt_2,99,63,915).  visible(wk2gt_2,915). happensAt(inactive(wk2gt_2),915). orientation(wk2gt_5,156,915). coords(wk2gt_5,64,96,915).  visible(wk2gt_5,915). happensAt(walking(wk2gt_5),915).
orientation(wk2gt_2,142,916). coords(wk2gt_2,99,63,916).  visible(wk2gt_2,916). happensAt(inactive(wk2gt_2),916). orientation(wk2gt_5,156,916). coords(wk2gt_5,65,96,916).  visible(wk2gt_5,916). happensAt(walking(wk2gt_5),916).
orientation(wk2gt_2,142,917). coords(wk2gt_2,99,63,917).  visible(wk2gt_2,917). happensAt(inactive(wk2gt_2),917). orientation(wk2gt_5,156,917). coords(wk2gt_5,66,97,917).  visible(wk2gt_5,917). happensAt(walking(wk2gt_5),917).
orientation(wk2gt_2,142,918). coords(wk2gt_2,99,63,918).  visible(wk2gt_2,918). happensAt(inactive(wk2gt_2),918). orientation(wk2gt_5,156,918). coords(wk2gt_5,66,97,918).  visible(wk2gt_5,918). happensAt(walking(wk2gt_5),918).
orientation(wk2gt_2,142,919). coords(wk2gt_2,99,63,919).  visible(wk2gt_2,919). happensAt(inactive(wk2gt_2),919). orientation(wk2gt_5,156,919). coords(wk2gt_5,66,97,919).  visible(wk2gt_5,919). happensAt(walking(wk2gt_5),919).
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
#show holdsAt/2.