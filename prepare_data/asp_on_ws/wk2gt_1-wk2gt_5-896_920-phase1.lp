
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

	orientation(wk2gt_1,127,896). coords(wk2gt_1,115,53,896).  visible(wk2gt_1,896). happensAt(inactive(wk2gt_1),896). orientation(wk2gt_5,156,896). coords(wk2gt_5,57,97,896).  visible(wk2gt_5,896). happensAt(walking(wk2gt_5),896).
orientation(wk2gt_1,127,897). coords(wk2gt_1,115,53,897).  visible(wk2gt_1,897). happensAt(inactive(wk2gt_1),897). orientation(wk2gt_5,156,897). coords(wk2gt_5,57,97,897).  visible(wk2gt_5,897). happensAt(walking(wk2gt_5),897).
orientation(wk2gt_1,127,898). coords(wk2gt_1,115,53,898).  visible(wk2gt_1,898). happensAt(inactive(wk2gt_1),898). orientation(wk2gt_5,156,898). coords(wk2gt_5,57,97,898).  visible(wk2gt_5,898). happensAt(walking(wk2gt_5),898).
orientation(wk2gt_1,127,899). coords(wk2gt_1,115,53,899).  visible(wk2gt_1,899). happensAt(inactive(wk2gt_1),899). orientation(wk2gt_5,156,899). coords(wk2gt_5,57,97,899).  visible(wk2gt_5,899). happensAt(walking(wk2gt_5),899).
orientation(wk2gt_1,127,900). coords(wk2gt_1,115,53,900).  visible(wk2gt_1,900). happensAt(inactive(wk2gt_1),900). orientation(wk2gt_5,156,900). coords(wk2gt_5,58,97,900).  visible(wk2gt_5,900). happensAt(walking(wk2gt_5),900).
orientation(wk2gt_1,127,901). coords(wk2gt_1,115,53,901).  visible(wk2gt_1,901). happensAt(inactive(wk2gt_1),901). orientation(wk2gt_5,156,901). coords(wk2gt_5,59,97,901).  visible(wk2gt_5,901). happensAt(walking(wk2gt_5),901).
orientation(wk2gt_1,127,902). coords(wk2gt_1,115,53,902).  visible(wk2gt_1,902). happensAt(inactive(wk2gt_1),902). orientation(wk2gt_5,156,902). coords(wk2gt_5,59,97,902).  visible(wk2gt_5,902). happensAt(walking(wk2gt_5),902).
orientation(wk2gt_1,127,903). coords(wk2gt_1,115,53,903).  visible(wk2gt_1,903). happensAt(inactive(wk2gt_1),903). orientation(wk2gt_5,156,903). coords(wk2gt_5,60,96,903).  visible(wk2gt_5,903). happensAt(walking(wk2gt_5),903).
orientation(wk2gt_1,127,904). coords(wk2gt_1,115,53,904).  visible(wk2gt_1,904). happensAt(inactive(wk2gt_1),904). orientation(wk2gt_5,156,904). coords(wk2gt_5,61,96,904).  visible(wk2gt_5,904). happensAt(walking(wk2gt_5),904).
orientation(wk2gt_1,127,905). coords(wk2gt_1,115,53,905).  visible(wk2gt_1,905). happensAt(inactive(wk2gt_1),905). orientation(wk2gt_5,156,905). coords(wk2gt_5,61,96,905).  visible(wk2gt_5,905). happensAt(walking(wk2gt_5),905).
orientation(wk2gt_1,127,906). coords(wk2gt_1,115,53,906).  visible(wk2gt_1,906). happensAt(inactive(wk2gt_1),906). orientation(wk2gt_5,156,906). coords(wk2gt_5,61,96,906).  visible(wk2gt_5,906). happensAt(walking(wk2gt_5),906).
orientation(wk2gt_1,127,907). coords(wk2gt_1,115,53,907).  visible(wk2gt_1,907). happensAt(inactive(wk2gt_1),907). orientation(wk2gt_5,156,907). coords(wk2gt_5,61,96,907).  visible(wk2gt_5,907). happensAt(walking(wk2gt_5),907).
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
#show holdsAt/2.