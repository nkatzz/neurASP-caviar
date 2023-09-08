
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

	orientation(wk2gt_1,124,908). coords(wk2gt_1,115,52,908).  visible(wk2gt_1,908). happensAt(inactive(wk2gt_1),908). orientation(wk2gt_5,156,908). coords(wk2gt_5,61,96,908).  visible(wk2gt_5,908). happensAt(walking(wk2gt_5),908).
orientation(wk2gt_1,127,909). coords(wk2gt_1,115,51,909).  visible(wk2gt_1,909). happensAt(inactive(wk2gt_1),909). orientation(wk2gt_5,156,909). coords(wk2gt_5,62,96,909).  visible(wk2gt_5,909). happensAt(walking(wk2gt_5),909).
orientation(wk2gt_1,127,910). coords(wk2gt_1,115,51,910).  visible(wk2gt_1,910). happensAt(inactive(wk2gt_1),910). orientation(wk2gt_5,156,910). coords(wk2gt_5,63,96,910).  visible(wk2gt_5,910). happensAt(walking(wk2gt_5),910).
orientation(wk2gt_1,127,911). coords(wk2gt_1,115,51,911).  visible(wk2gt_1,911). happensAt(inactive(wk2gt_1),911). orientation(wk2gt_5,156,911). coords(wk2gt_5,63,96,911).  visible(wk2gt_5,911). happensAt(walking(wk2gt_5),911).
orientation(wk2gt_1,127,912). coords(wk2gt_1,115,51,912).  visible(wk2gt_1,912). happensAt(inactive(wk2gt_1),912). orientation(wk2gt_5,156,912). coords(wk2gt_5,63,96,912).  visible(wk2gt_5,912). happensAt(walking(wk2gt_5),912).
orientation(wk2gt_1,127,913). coords(wk2gt_1,115,51,913).  visible(wk2gt_1,913). happensAt(inactive(wk2gt_1),913). orientation(wk2gt_5,156,913). coords(wk2gt_5,63,96,913).  visible(wk2gt_5,913). happensAt(walking(wk2gt_5),913).
orientation(wk2gt_1,127,914). coords(wk2gt_1,115,51,914).  visible(wk2gt_1,914). happensAt(inactive(wk2gt_1),914). orientation(wk2gt_5,156,914). coords(wk2gt_5,63,96,914).  visible(wk2gt_5,914). happensAt(walking(wk2gt_5),914).
orientation(wk2gt_1,127,915). coords(wk2gt_1,115,51,915).  visible(wk2gt_1,915). happensAt(inactive(wk2gt_1),915). orientation(wk2gt_5,156,915). coords(wk2gt_5,64,96,915).  visible(wk2gt_5,915). happensAt(walking(wk2gt_5),915).
orientation(wk2gt_1,127,916). coords(wk2gt_1,116,51,916).  visible(wk2gt_1,916). happensAt(inactive(wk2gt_1),916). orientation(wk2gt_5,156,916). coords(wk2gt_5,65,96,916).  visible(wk2gt_5,916). happensAt(walking(wk2gt_5),916).
orientation(wk2gt_1,132,917). coords(wk2gt_1,116,51,917).  visible(wk2gt_1,917). happensAt(inactive(wk2gt_1),917). orientation(wk2gt_5,156,917). coords(wk2gt_5,66,97,917).  visible(wk2gt_5,917). happensAt(walking(wk2gt_5),917).
orientation(wk2gt_1,132,918). coords(wk2gt_1,117,51,918).  visible(wk2gt_1,918). happensAt(inactive(wk2gt_1),918). orientation(wk2gt_5,156,918). coords(wk2gt_5,66,97,918).  visible(wk2gt_5,918). happensAt(walking(wk2gt_5),918).
orientation(wk2gt_1,132,919). coords(wk2gt_1,117,51,919).  visible(wk2gt_1,919). happensAt(inactive(wk2gt_1),919). orientation(wk2gt_5,156,919). coords(wk2gt_5,66,97,919).  visible(wk2gt_5,919). happensAt(walking(wk2gt_5),919).
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
#show holdsAt/2.