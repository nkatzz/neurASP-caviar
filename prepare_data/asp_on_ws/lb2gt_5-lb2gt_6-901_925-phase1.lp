
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

	orientation(lb2gt_5,152,901). coords(lb2gt_5,71,113,901).  visible(lb2gt_5,901). happensAt(walking(lb2gt_5),901). orientation(lb2gt_6,151,901). coords(lb2gt_6,71,122,901).  visible(lb2gt_6,901). happensAt(inactive(lb2gt_6),901).
orientation(lb2gt_5,152,902). coords(lb2gt_5,71,112,902).  visible(lb2gt_5,902). happensAt(walking(lb2gt_5),902). orientation(lb2gt_6,151,902). coords(lb2gt_6,71,122,902).  visible(lb2gt_6,902). happensAt(inactive(lb2gt_6),902).
orientation(lb2gt_5,152,903). coords(lb2gt_5,71,113,903).  visible(lb2gt_5,903). happensAt(walking(lb2gt_5),903). orientation(lb2gt_6,151,903). coords(lb2gt_6,71,122,903).  visible(lb2gt_6,903). happensAt(inactive(lb2gt_6),903).
orientation(lb2gt_5,152,904). coords(lb2gt_5,71,113,904).  visible(lb2gt_5,904). happensAt(walking(lb2gt_5),904). orientation(lb2gt_6,151,904). coords(lb2gt_6,71,122,904).  visible(lb2gt_6,904). happensAt(inactive(lb2gt_6),904).
orientation(lb2gt_5,152,905). coords(lb2gt_5,71,113,905).  visible(lb2gt_5,905). happensAt(walking(lb2gt_5),905). orientation(lb2gt_6,151,905). coords(lb2gt_6,71,121,905).  visible(lb2gt_6,905). happensAt(inactive(lb2gt_6),905).
orientation(lb2gt_5,152,906). coords(lb2gt_5,71,113,906).  visible(lb2gt_5,906). happensAt(walking(lb2gt_5),906). orientation(lb2gt_6,151,906). coords(lb2gt_6,71,121,906).  visible(lb2gt_6,906). happensAt(inactive(lb2gt_6),906).
orientation(lb2gt_5,152,907). coords(lb2gt_5,72,113,907).  visible(lb2gt_5,907). happensAt(walking(lb2gt_5),907). orientation(lb2gt_6,151,907). coords(lb2gt_6,71,121,907).  visible(lb2gt_6,907). happensAt(inactive(lb2gt_6),907).
orientation(lb2gt_5,152,908). coords(lb2gt_5,72,112,908).  visible(lb2gt_5,908). happensAt(walking(lb2gt_5),908). orientation(lb2gt_6,151,908). coords(lb2gt_6,71,121,908).  visible(lb2gt_6,908). happensAt(inactive(lb2gt_6),908).
orientation(lb2gt_5,152,909). coords(lb2gt_5,72,112,909).  visible(lb2gt_5,909). happensAt(walking(lb2gt_5),909). orientation(lb2gt_6,151,909). coords(lb2gt_6,71,121,909).  visible(lb2gt_6,909). happensAt(inactive(lb2gt_6),909).
orientation(lb2gt_5,152,910). coords(lb2gt_5,72,112,910).  visible(lb2gt_5,910). happensAt(walking(lb2gt_5),910). orientation(lb2gt_6,151,910). coords(lb2gt_6,71,121,910).  visible(lb2gt_6,910). happensAt(inactive(lb2gt_6),910).
orientation(lb2gt_5,152,911). coords(lb2gt_5,72,112,911).  visible(lb2gt_5,911). happensAt(walking(lb2gt_5),911). orientation(lb2gt_6,151,911). coords(lb2gt_6,71,121,911).  visible(lb2gt_6,911). happensAt(inactive(lb2gt_6),911).
orientation(lb2gt_5,152,912). coords(lb2gt_5,72,112,912).  visible(lb2gt_5,912). happensAt(walking(lb2gt_5),912). orientation(lb2gt_6,151,912). coords(lb2gt_6,72,121,912).  visible(lb2gt_6,912). happensAt(inactive(lb2gt_6),912).
orientation(lb2gt_5,152,913). coords(lb2gt_5,72,112,913).  visible(lb2gt_5,913). happensAt(walking(lb2gt_5),913). orientation(lb2gt_6,151,913). coords(lb2gt_6,72,121,913).  visible(lb2gt_6,913). happensAt(inactive(lb2gt_6),913).
orientation(lb2gt_5,152,914). coords(lb2gt_5,73,112,914).  visible(lb2gt_5,914). happensAt(walking(lb2gt_5),914). orientation(lb2gt_6,151,914). coords(lb2gt_6,73,122,914).  visible(lb2gt_6,914). happensAt(inactive(lb2gt_6),914).
orientation(lb2gt_5,152,915). coords(lb2gt_5,73,112,915).  visible(lb2gt_5,915). happensAt(walking(lb2gt_5),915). orientation(lb2gt_6,151,915). coords(lb2gt_6,73,122,915).  visible(lb2gt_6,915). happensAt(inactive(lb2gt_6),915).
orientation(lb2gt_5,152,916). coords(lb2gt_5,73,113,916).  visible(lb2gt_5,916). happensAt(walking(lb2gt_5),916). orientation(lb2gt_6,151,916). coords(lb2gt_6,74,122,916).  visible(lb2gt_6,916). happensAt(inactive(lb2gt_6),916).
orientation(lb2gt_5,152,917). coords(lb2gt_5,73,113,917).  visible(lb2gt_5,917). happensAt(walking(lb2gt_5),917). orientation(lb2gt_6,151,917). coords(lb2gt_6,75,122,917).  visible(lb2gt_6,917). happensAt(inactive(lb2gt_6),917).
orientation(lb2gt_5,152,918). coords(lb2gt_5,74,114,918).  visible(lb2gt_5,918). happensAt(walking(lb2gt_5),918). orientation(lb2gt_6,151,918). coords(lb2gt_6,75,122,918).  visible(lb2gt_6,918). happensAt(inactive(lb2gt_6),918).
orientation(lb2gt_5,152,919). coords(lb2gt_5,75,114,919).  visible(lb2gt_5,919). happensAt(walking(lb2gt_5),919). orientation(lb2gt_6,151,919). coords(lb2gt_6,76,122,919).  visible(lb2gt_6,919). happensAt(inactive(lb2gt_6),919).
orientation(lb2gt_5,152,920). coords(lb2gt_5,76,113,920).  visible(lb2gt_5,920). happensAt(walking(lb2gt_5),920). orientation(lb2gt_6,151,920). coords(lb2gt_6,77,122,920).  visible(lb2gt_6,920). happensAt(inactive(lb2gt_6),920).
orientation(lb2gt_5,152,921). coords(lb2gt_5,76,114,921).  visible(lb2gt_5,921). happensAt(walking(lb2gt_5),921). orientation(lb2gt_6,151,921). coords(lb2gt_6,77,122,921).  visible(lb2gt_6,921). happensAt(inactive(lb2gt_6),921).
orientation(lb2gt_5,152,922). coords(lb2gt_5,76,114,922).  visible(lb2gt_5,922). happensAt(walking(lb2gt_5),922). orientation(lb2gt_6,151,922). coords(lb2gt_6,78,122,922).  visible(lb2gt_6,922). happensAt(inactive(lb2gt_6),922).
orientation(lb2gt_5,152,923). coords(lb2gt_5,77,115,923).  visible(lb2gt_5,923). happensAt(walking(lb2gt_5),923). orientation(lb2gt_6,151,923). coords(lb2gt_6,79,122,923).  visible(lb2gt_6,923). happensAt(inactive(lb2gt_6),923).
orientation(lb2gt_5,152,924). coords(lb2gt_5,77,115,924).  visible(lb2gt_5,924). happensAt(walking(lb2gt_5),924). orientation(lb2gt_6,151,924). coords(lb2gt_6,80,122,924).  visible(lb2gt_6,924). happensAt(inactive(lb2gt_6),924).
orientation(lb2gt_5,152,925). coords(lb2gt_5,77,115,925).  visible(lb2gt_5,925). happensAt(walking(lb2gt_5),925). orientation(lb2gt_6,151,925). coords(lb2gt_6,80,123,925).  happensAt(disappear(lb2gt_6),925). happensAt(inactive(lb2gt_6),925).
#show holdsAt/2.