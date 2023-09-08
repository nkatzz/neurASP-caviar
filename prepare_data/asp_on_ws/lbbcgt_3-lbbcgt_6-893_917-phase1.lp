
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

	orientation(lbbcgt_3,163,893). coords(lbbcgt_3,39,148,893).  visible(lbbcgt_3,893). happensAt(walking(lbbcgt_3),893). orientation(lbbcgt_6,142,893). coords(lbbcgt_6,134,64,893).  visible(lbbcgt_6,893). happensAt(walking(lbbcgt_6),893).
orientation(lbbcgt_3,163,894). coords(lbbcgt_3,39,149,894).  visible(lbbcgt_3,894). happensAt(walking(lbbcgt_3),894). orientation(lbbcgt_6,142,894). coords(lbbcgt_6,133,66,894).  visible(lbbcgt_6,894). happensAt(walking(lbbcgt_6),894).
orientation(lbbcgt_3,163,895). coords(lbbcgt_3,39,149,895).  visible(lbbcgt_3,895). happensAt(walking(lbbcgt_3),895). orientation(lbbcgt_6,142,895). coords(lbbcgt_6,133,67,895).  visible(lbbcgt_6,895). happensAt(walking(lbbcgt_6),895).
orientation(lbbcgt_3,163,896). coords(lbbcgt_3,39,149,896).  visible(lbbcgt_3,896). happensAt(walking(lbbcgt_3),896). orientation(lbbcgt_6,142,896). coords(lbbcgt_6,132,68,896).  visible(lbbcgt_6,896). happensAt(walking(lbbcgt_6),896).
orientation(lbbcgt_3,163,897). coords(lbbcgt_3,39,150,897).  visible(lbbcgt_3,897). happensAt(walking(lbbcgt_3),897). orientation(lbbcgt_6,142,897). coords(lbbcgt_6,132,68,897).  visible(lbbcgt_6,897). happensAt(walking(lbbcgt_6),897).
orientation(lbbcgt_3,163,898). coords(lbbcgt_3,39,151,898).  visible(lbbcgt_3,898). happensAt(walking(lbbcgt_3),898). orientation(lbbcgt_6,142,898). coords(lbbcgt_6,131,69,898).  visible(lbbcgt_6,898). happensAt(walking(lbbcgt_6),898).
orientation(lbbcgt_3,163,899). coords(lbbcgt_3,39,151,899).  visible(lbbcgt_3,899). happensAt(walking(lbbcgt_3),899). orientation(lbbcgt_6,142,899). coords(lbbcgt_6,129,70,899).  visible(lbbcgt_6,899). happensAt(walking(lbbcgt_6),899).
orientation(lbbcgt_3,163,900). coords(lbbcgt_3,39,151,900).  visible(lbbcgt_3,900). happensAt(walking(lbbcgt_3),900). orientation(lbbcgt_6,142,900). coords(lbbcgt_6,127,70,900).  visible(lbbcgt_6,900). happensAt(walking(lbbcgt_6),900).
orientation(lbbcgt_3,163,901). coords(lbbcgt_3,39,152,901).  visible(lbbcgt_3,901). happensAt(walking(lbbcgt_3),901). orientation(lbbcgt_6,142,901). coords(lbbcgt_6,125,70,901).  visible(lbbcgt_6,901). happensAt(walking(lbbcgt_6),901).
orientation(lbbcgt_3,163,902). coords(lbbcgt_3,39,152,902).  visible(lbbcgt_3,902). happensAt(walking(lbbcgt_3),902). orientation(lbbcgt_6,142,902). coords(lbbcgt_6,123,71,902).  visible(lbbcgt_6,902). happensAt(walking(lbbcgt_6),902).
orientation(lbbcgt_3,163,903). coords(lbbcgt_3,39,153,903).  visible(lbbcgt_3,903). happensAt(walking(lbbcgt_3),903). orientation(lbbcgt_6,142,903). coords(lbbcgt_6,122,71,903).  visible(lbbcgt_6,903). happensAt(walking(lbbcgt_6),903).
orientation(lbbcgt_3,163,904). coords(lbbcgt_3,38,155,904).  visible(lbbcgt_3,904). happensAt(walking(lbbcgt_3),904). orientation(lbbcgt_6,142,904). coords(lbbcgt_6,121,71,904).  visible(lbbcgt_6,904). happensAt(walking(lbbcgt_6),904).
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
#show holdsAt/2.