
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

	orientation(lbbcgt_5,21,897). coords(lbbcgt_5,231,259,897).  visible(lbbcgt_5,897). happensAt(walking(lbbcgt_5),897). orientation(lbbcgt_6,142,897). coords(lbbcgt_6,132,68,897).  visible(lbbcgt_6,897). happensAt(walking(lbbcgt_6),897).
orientation(lbbcgt_5,9,898). coords(lbbcgt_5,230,257,898).  visible(lbbcgt_5,898). happensAt(walking(lbbcgt_5),898). orientation(lbbcgt_6,142,898). coords(lbbcgt_6,131,69,898).  visible(lbbcgt_6,898). happensAt(walking(lbbcgt_6),898).
orientation(lbbcgt_5,9,899). coords(lbbcgt_5,229,256,899).  visible(lbbcgt_5,899). happensAt(walking(lbbcgt_5),899). orientation(lbbcgt_6,142,899). coords(lbbcgt_6,129,70,899).  visible(lbbcgt_6,899). happensAt(walking(lbbcgt_6),899).
orientation(lbbcgt_5,9,900). coords(lbbcgt_5,229,255,900).  visible(lbbcgt_5,900). happensAt(walking(lbbcgt_5),900). orientation(lbbcgt_6,142,900). coords(lbbcgt_6,127,70,900).  visible(lbbcgt_6,900). happensAt(walking(lbbcgt_6),900).
orientation(lbbcgt_5,9,901). coords(lbbcgt_5,229,253,901).  visible(lbbcgt_5,901). happensAt(walking(lbbcgt_5),901). orientation(lbbcgt_6,142,901). coords(lbbcgt_6,125,70,901).  visible(lbbcgt_6,901). happensAt(walking(lbbcgt_6),901).
orientation(lbbcgt_5,9,902). coords(lbbcgt_5,228,251,902).  visible(lbbcgt_5,902). happensAt(walking(lbbcgt_5),902). orientation(lbbcgt_6,142,902). coords(lbbcgt_6,123,71,902).  visible(lbbcgt_6,902). happensAt(walking(lbbcgt_6),902).
orientation(lbbcgt_5,9,903). coords(lbbcgt_5,228,247,903).  visible(lbbcgt_5,903). happensAt(walking(lbbcgt_5),903). orientation(lbbcgt_6,142,903). coords(lbbcgt_6,122,71,903).  visible(lbbcgt_6,903). happensAt(walking(lbbcgt_6),903).
orientation(lbbcgt_5,9,904). coords(lbbcgt_5,227,244,904).  visible(lbbcgt_5,904). happensAt(walking(lbbcgt_5),904). orientation(lbbcgt_6,142,904). coords(lbbcgt_6,121,71,904).  visible(lbbcgt_6,904). happensAt(walking(lbbcgt_6),904).
orientation(lbbcgt_5,9,905). coords(lbbcgt_5,227,242,905).  visible(lbbcgt_5,905). happensAt(walking(lbbcgt_5),905). orientation(lbbcgt_6,142,905). coords(lbbcgt_6,121,72,905).  visible(lbbcgt_6,905). happensAt(walking(lbbcgt_6),905).
orientation(lbbcgt_5,9,906). coords(lbbcgt_5,226,240,906).  visible(lbbcgt_5,906). happensAt(walking(lbbcgt_5),906). orientation(lbbcgt_6,142,906). coords(lbbcgt_6,120,73,906).  visible(lbbcgt_6,906). happensAt(walking(lbbcgt_6),906).
orientation(lbbcgt_5,9,907). coords(lbbcgt_5,225,240,907).  visible(lbbcgt_5,907). happensAt(walking(lbbcgt_5),907). orientation(lbbcgt_6,142,907). coords(lbbcgt_6,120,73,907).  visible(lbbcgt_6,907). happensAt(walking(lbbcgt_6),907).
orientation(lbbcgt_5,9,908). coords(lbbcgt_5,225,240,908).  visible(lbbcgt_5,908). happensAt(walking(lbbcgt_5),908). orientation(lbbcgt_6,142,908). coords(lbbcgt_6,120,74,908).  visible(lbbcgt_6,908). happensAt(walking(lbbcgt_6),908).
orientation(lbbcgt_5,9,909). coords(lbbcgt_5,224,239,909).  visible(lbbcgt_5,909). happensAt(walking(lbbcgt_5),909). orientation(lbbcgt_6,142,909). coords(lbbcgt_6,118,74,909).  visible(lbbcgt_6,909). happensAt(walking(lbbcgt_6),909).
orientation(lbbcgt_5,9,910). coords(lbbcgt_5,223,238,910).  visible(lbbcgt_5,910). happensAt(walking(lbbcgt_5),910). orientation(lbbcgt_6,142,910). coords(lbbcgt_6,118,76,910).  visible(lbbcgt_6,910). happensAt(walking(lbbcgt_6),910).
orientation(lbbcgt_5,9,911). coords(lbbcgt_5,221,236,911).  visible(lbbcgt_5,911). happensAt(walking(lbbcgt_5),911). orientation(lbbcgt_6,142,911). coords(lbbcgt_6,117,75,911).  visible(lbbcgt_6,911). happensAt(walking(lbbcgt_6),911).
orientation(lbbcgt_5,9,912). coords(lbbcgt_5,220,235,912).  visible(lbbcgt_5,912). happensAt(walking(lbbcgt_5),912). orientation(lbbcgt_6,144,912). coords(lbbcgt_6,115,76,912).  visible(lbbcgt_6,912). happensAt(walking(lbbcgt_6),912).
orientation(lbbcgt_5,9,913). coords(lbbcgt_5,218,230,913).  visible(lbbcgt_5,913). happensAt(walking(lbbcgt_5),913). orientation(lbbcgt_6,144,913). coords(lbbcgt_6,113,76,913).  visible(lbbcgt_6,913). happensAt(walking(lbbcgt_6),913).
orientation(lbbcgt_5,9,914). coords(lbbcgt_5,217,227,914).  visible(lbbcgt_5,914). happensAt(walking(lbbcgt_5),914). orientation(lbbcgt_6,144,914). coords(lbbcgt_6,112,76,914).  visible(lbbcgt_6,914). happensAt(walking(lbbcgt_6),914).
orientation(lbbcgt_5,9,915). coords(lbbcgt_5,215,225,915).  visible(lbbcgt_5,915). happensAt(walking(lbbcgt_5),915). orientation(lbbcgt_6,144,915). coords(lbbcgt_6,110,77,915).  visible(lbbcgt_6,915). happensAt(walking(lbbcgt_6),915).
orientation(lbbcgt_5,9,916). coords(lbbcgt_5,214,222,916).  visible(lbbcgt_5,916). happensAt(walking(lbbcgt_5),916). orientation(lbbcgt_6,144,916). coords(lbbcgt_6,110,77,916).  visible(lbbcgt_6,916). happensAt(walking(lbbcgt_6),916).
orientation(lbbcgt_5,9,917). coords(lbbcgt_5,212,219,917).  visible(lbbcgt_5,917). happensAt(walking(lbbcgt_5),917). orientation(lbbcgt_6,144,917). coords(lbbcgt_6,109,79,917).  visible(lbbcgt_6,917). happensAt(walking(lbbcgt_6),917).
orientation(lbbcgt_5,9,918). coords(lbbcgt_5,212,217,918).  visible(lbbcgt_5,918). happensAt(walking(lbbcgt_5),918). orientation(lbbcgt_6,144,918). coords(lbbcgt_6,108,79,918).  visible(lbbcgt_6,918). happensAt(walking(lbbcgt_6),918).
orientation(lbbcgt_5,173,919). coords(lbbcgt_5,212,214,919).  visible(lbbcgt_5,919). happensAt(walking(lbbcgt_5),919). orientation(lbbcgt_6,144,919). coords(lbbcgt_6,107,80,919).  visible(lbbcgt_6,919). happensAt(walking(lbbcgt_6),919).
orientation(lbbcgt_5,167,920). coords(lbbcgt_5,212,213,920).  visible(lbbcgt_5,920). happensAt(walking(lbbcgt_5),920). orientation(lbbcgt_6,144,920). coords(lbbcgt_6,107,81,920).  visible(lbbcgt_6,920). happensAt(walking(lbbcgt_6),920).
orientation(lbbcgt_5,167,921). coords(lbbcgt_5,212,213,921).  visible(lbbcgt_5,921). happensAt(walking(lbbcgt_5),921). orientation(lbbcgt_6,144,921). coords(lbbcgt_6,107,81,921).  visible(lbbcgt_6,921). happensAt(walking(lbbcgt_6),921).
#show holdsAt/2.