
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

	orientation(wk2gt_2,142,896). coords(wk2gt_2,99,63,896).  visible(wk2gt_2,896). happensAt(inactive(wk2gt_2),896). orientation(wk2gt_5,156,896). coords(wk2gt_5,57,97,896).  visible(wk2gt_5,896). happensAt(walking(wk2gt_5),896).
orientation(wk2gt_2,142,897). coords(wk2gt_2,99,63,897).  visible(wk2gt_2,897). happensAt(inactive(wk2gt_2),897). orientation(wk2gt_5,156,897). coords(wk2gt_5,57,97,897).  visible(wk2gt_5,897). happensAt(walking(wk2gt_5),897).
orientation(wk2gt_2,142,898). coords(wk2gt_2,99,63,898).  visible(wk2gt_2,898). happensAt(inactive(wk2gt_2),898). orientation(wk2gt_5,156,898). coords(wk2gt_5,57,97,898).  visible(wk2gt_5,898). happensAt(walking(wk2gt_5),898).
orientation(wk2gt_2,142,899). coords(wk2gt_2,99,63,899).  visible(wk2gt_2,899). happensAt(inactive(wk2gt_2),899). orientation(wk2gt_5,156,899). coords(wk2gt_5,57,97,899).  visible(wk2gt_5,899). happensAt(walking(wk2gt_5),899).
orientation(wk2gt_2,142,900). coords(wk2gt_2,99,63,900).  visible(wk2gt_2,900). happensAt(inactive(wk2gt_2),900). orientation(wk2gt_5,156,900). coords(wk2gt_5,58,97,900).  visible(wk2gt_5,900). happensAt(walking(wk2gt_5),900).
orientation(wk2gt_2,142,901). coords(wk2gt_2,99,63,901).  visible(wk2gt_2,901). happensAt(inactive(wk2gt_2),901). orientation(wk2gt_5,156,901). coords(wk2gt_5,59,97,901).  visible(wk2gt_5,901). happensAt(walking(wk2gt_5),901).
orientation(wk2gt_2,142,902). coords(wk2gt_2,99,63,902).  visible(wk2gt_2,902). happensAt(inactive(wk2gt_2),902). orientation(wk2gt_5,156,902). coords(wk2gt_5,59,97,902).  visible(wk2gt_5,902). happensAt(walking(wk2gt_5),902).
orientation(wk2gt_2,142,903). coords(wk2gt_2,99,63,903).  visible(wk2gt_2,903). happensAt(inactive(wk2gt_2),903). orientation(wk2gt_5,156,903). coords(wk2gt_5,60,96,903).  visible(wk2gt_5,903). happensAt(walking(wk2gt_5),903).
orientation(wk2gt_2,142,904). coords(wk2gt_2,99,63,904).  visible(wk2gt_2,904). happensAt(inactive(wk2gt_2),904). orientation(wk2gt_5,156,904). coords(wk2gt_5,61,96,904).  visible(wk2gt_5,904). happensAt(walking(wk2gt_5),904).
orientation(wk2gt_2,142,905). coords(wk2gt_2,99,63,905).  visible(wk2gt_2,905). happensAt(inactive(wk2gt_2),905). orientation(wk2gt_5,156,905). coords(wk2gt_5,61,96,905).  visible(wk2gt_5,905). happensAt(walking(wk2gt_5),905).
orientation(wk2gt_2,142,906). coords(wk2gt_2,99,63,906).  visible(wk2gt_2,906). happensAt(inactive(wk2gt_2),906). orientation(wk2gt_5,156,906). coords(wk2gt_5,61,96,906).  visible(wk2gt_5,906). happensAt(walking(wk2gt_5),906).
orientation(wk2gt_2,142,907). coords(wk2gt_2,99,63,907).  visible(wk2gt_2,907). happensAt(inactive(wk2gt_2),907). orientation(wk2gt_5,156,907). coords(wk2gt_5,61,96,907).  visible(wk2gt_5,907). happensAt(walking(wk2gt_5),907).
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
#show holdsAt/2.