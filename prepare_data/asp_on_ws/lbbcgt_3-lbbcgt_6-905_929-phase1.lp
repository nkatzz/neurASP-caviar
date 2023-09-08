
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

	orientation(lbbcgt_3,163,905). coords(lbbcgt_3,38,157,905).  visible(lbbcgt_3,905). happensAt(walking(lbbcgt_3),905). orientation(lbbcgt_6,142,905). coords(lbbcgt_6,121,72,905).  visible(lbbcgt_6,905). happensAt(walking(lbbcgt_6),905).
orientation(lbbcgt_3,163,906). coords(lbbcgt_3,38,158,906).  visible(lbbcgt_3,906). happensAt(walking(lbbcgt_3),906). orientation(lbbcgt_6,142,906). coords(lbbcgt_6,120,73,906).  visible(lbbcgt_6,906). happensAt(walking(lbbcgt_6),906).
orientation(lbbcgt_3,163,907). coords(lbbcgt_3,38,158,907).  visible(lbbcgt_3,907). happensAt(walking(lbbcgt_3),907). orientation(lbbcgt_6,142,907). coords(lbbcgt_6,120,73,907).  visible(lbbcgt_6,907). happensAt(walking(lbbcgt_6),907).
orientation(lbbcgt_3,163,908). coords(lbbcgt_3,38,159,908).  visible(lbbcgt_3,908). happensAt(walking(lbbcgt_3),908). orientation(lbbcgt_6,142,908). coords(lbbcgt_6,120,74,908).  visible(lbbcgt_6,908). happensAt(walking(lbbcgt_6),908).
orientation(lbbcgt_3,163,909). coords(lbbcgt_3,38,159,909).  visible(lbbcgt_3,909). happensAt(walking(lbbcgt_3),909). orientation(lbbcgt_6,142,909). coords(lbbcgt_6,118,74,909).  visible(lbbcgt_6,909). happensAt(walking(lbbcgt_6),909).
orientation(lbbcgt_3,163,910). coords(lbbcgt_3,37,159,910).  visible(lbbcgt_3,910). happensAt(walking(lbbcgt_3),910). orientation(lbbcgt_6,142,910). coords(lbbcgt_6,118,76,910).  visible(lbbcgt_6,910). happensAt(walking(lbbcgt_6),910).
orientation(lbbcgt_3,163,911). coords(lbbcgt_3,37,160,911).  visible(lbbcgt_3,911). happensAt(walking(lbbcgt_3),911). orientation(lbbcgt_6,142,911). coords(lbbcgt_6,117,75,911).  visible(lbbcgt_6,911). happensAt(walking(lbbcgt_6),911).
orientation(lbbcgt_3,163,912). coords(lbbcgt_3,37,160,912).  visible(lbbcgt_3,912). happensAt(walking(lbbcgt_3),912). orientation(lbbcgt_6,144,912). coords(lbbcgt_6,115,76,912).  visible(lbbcgt_6,912). happensAt(walking(lbbcgt_6),912).
orientation(lbbcgt_3,163,913). coords(lbbcgt_3,37,160,913).  visible(lbbcgt_3,913). happensAt(walking(lbbcgt_3),913). orientation(lbbcgt_6,144,913). coords(lbbcgt_6,113,76,913).  visible(lbbcgt_6,913). happensAt(walking(lbbcgt_6),913).
orientation(lbbcgt_3,163,914). coords(lbbcgt_3,37,161,914).  visible(lbbcgt_3,914). happensAt(walking(lbbcgt_3),914). orientation(lbbcgt_6,144,914). coords(lbbcgt_6,112,76,914).  visible(lbbcgt_6,914). happensAt(walking(lbbcgt_6),914).
orientation(lbbcgt_3,163,915). coords(lbbcgt_3,37,161,915).  visible(lbbcgt_3,915). happensAt(walking(lbbcgt_3),915). orientation(lbbcgt_6,144,915). coords(lbbcgt_6,110,77,915).  visible(lbbcgt_6,915). happensAt(walking(lbbcgt_6),915).
orientation(lbbcgt_3,163,916). coords(lbbcgt_3,37,161,916).  visible(lbbcgt_3,916). happensAt(walking(lbbcgt_3),916). orientation(lbbcgt_6,144,916). coords(lbbcgt_6,110,77,916).  visible(lbbcgt_6,916). happensAt(walking(lbbcgt_6),916).
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
#show holdsAt/2.