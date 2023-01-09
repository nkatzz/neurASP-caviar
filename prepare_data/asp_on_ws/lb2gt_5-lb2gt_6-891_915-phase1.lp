
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

	orientation(lb2gt_5,152,891). coords(lb2gt_5,72,112,891).  visible(lb2gt_5,891). happensAt(walking(lb2gt_5),891). orientation(lb2gt_6,151,891). coords(lb2gt_6,72,122,891).  visible(lb2gt_6,891). happensAt(inactive(lb2gt_6),891).
orientation(lb2gt_5,152,892). coords(lb2gt_5,72,112,892).  visible(lb2gt_5,892). happensAt(walking(lb2gt_5),892). orientation(lb2gt_6,151,892). coords(lb2gt_6,72,122,892).  visible(lb2gt_6,892). happensAt(inactive(lb2gt_6),892).
orientation(lb2gt_5,152,893). coords(lb2gt_5,71,112,893).  visible(lb2gt_5,893). happensAt(walking(lb2gt_5),893). orientation(lb2gt_6,151,893). coords(lb2gt_6,72,122,893).  visible(lb2gt_6,893). happensAt(inactive(lb2gt_6),893).
orientation(lb2gt_5,152,894). coords(lb2gt_5,71,112,894).  visible(lb2gt_5,894). happensAt(walking(lb2gt_5),894). orientation(lb2gt_6,151,894). coords(lb2gt_6,72,122,894).  visible(lb2gt_6,894). happensAt(inactive(lb2gt_6),894).
orientation(lb2gt_5,152,895). coords(lb2gt_5,71,113,895).  visible(lb2gt_5,895). happensAt(walking(lb2gt_5),895). orientation(lb2gt_6,151,895). coords(lb2gt_6,72,122,895).  visible(lb2gt_6,895). happensAt(inactive(lb2gt_6),895).
orientation(lb2gt_5,152,896). coords(lb2gt_5,71,113,896).  visible(lb2gt_5,896). happensAt(walking(lb2gt_5),896). orientation(lb2gt_6,151,896). coords(lb2gt_6,72,122,896).  visible(lb2gt_6,896). happensAt(inactive(lb2gt_6),896).
orientation(lb2gt_5,152,897). coords(lb2gt_5,71,112,897).  visible(lb2gt_5,897). happensAt(walking(lb2gt_5),897). orientation(lb2gt_6,151,897). coords(lb2gt_6,72,122,897).  visible(lb2gt_6,897). happensAt(inactive(lb2gt_6),897).
orientation(lb2gt_5,152,898). coords(lb2gt_5,71,113,898).  visible(lb2gt_5,898). happensAt(walking(lb2gt_5),898). orientation(lb2gt_6,151,898). coords(lb2gt_6,72,122,898).  visible(lb2gt_6,898). happensAt(inactive(lb2gt_6),898).
orientation(lb2gt_5,152,899). coords(lb2gt_5,71,113,899).  visible(lb2gt_5,899). happensAt(walking(lb2gt_5),899). orientation(lb2gt_6,151,899). coords(lb2gt_6,72,122,899).  visible(lb2gt_6,899). happensAt(inactive(lb2gt_6),899).
orientation(lb2gt_5,152,900). coords(lb2gt_5,70,113,900).  visible(lb2gt_5,900). happensAt(walking(lb2gt_5),900). orientation(lb2gt_6,151,900). coords(lb2gt_6,72,122,900).  visible(lb2gt_6,900). happensAt(inactive(lb2gt_6),900).
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
#show holdsAt/2.