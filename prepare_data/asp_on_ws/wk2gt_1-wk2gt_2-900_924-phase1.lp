
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

	orientation(wk2gt_1,127,900). coords(wk2gt_1,115,53,900).  visible(wk2gt_1,900). happensAt(inactive(wk2gt_1),900). orientation(wk2gt_2,142,900). coords(wk2gt_2,99,63,900).  visible(wk2gt_2,900). happensAt(inactive(wk2gt_2),900).
orientation(wk2gt_1,127,901). coords(wk2gt_1,115,53,901).  visible(wk2gt_1,901). happensAt(inactive(wk2gt_1),901). orientation(wk2gt_2,142,901). coords(wk2gt_2,99,63,901).  visible(wk2gt_2,901). happensAt(inactive(wk2gt_2),901).
orientation(wk2gt_1,127,902). coords(wk2gt_1,115,53,902).  visible(wk2gt_1,902). happensAt(inactive(wk2gt_1),902). orientation(wk2gt_2,142,902). coords(wk2gt_2,99,63,902).  visible(wk2gt_2,902). happensAt(inactive(wk2gt_2),902).
orientation(wk2gt_1,127,903). coords(wk2gt_1,115,53,903).  visible(wk2gt_1,903). happensAt(inactive(wk2gt_1),903). orientation(wk2gt_2,142,903). coords(wk2gt_2,99,63,903).  visible(wk2gt_2,903). happensAt(inactive(wk2gt_2),903).
orientation(wk2gt_1,127,904). coords(wk2gt_1,115,53,904).  visible(wk2gt_1,904). happensAt(inactive(wk2gt_1),904). orientation(wk2gt_2,142,904). coords(wk2gt_2,99,63,904).  visible(wk2gt_2,904). happensAt(inactive(wk2gt_2),904).
orientation(wk2gt_1,127,905). coords(wk2gt_1,115,53,905).  visible(wk2gt_1,905). happensAt(inactive(wk2gt_1),905). orientation(wk2gt_2,142,905). coords(wk2gt_2,99,63,905).  visible(wk2gt_2,905). happensAt(inactive(wk2gt_2),905).
orientation(wk2gt_1,127,906). coords(wk2gt_1,115,53,906).  visible(wk2gt_1,906). happensAt(inactive(wk2gt_1),906). orientation(wk2gt_2,142,906). coords(wk2gt_2,99,63,906).  visible(wk2gt_2,906). happensAt(inactive(wk2gt_2),906).
orientation(wk2gt_1,127,907). coords(wk2gt_1,115,53,907).  visible(wk2gt_1,907). happensAt(inactive(wk2gt_1),907). orientation(wk2gt_2,142,907). coords(wk2gt_2,99,63,907).  visible(wk2gt_2,907). happensAt(inactive(wk2gt_2),907).
orientation(wk2gt_1,124,908). coords(wk2gt_1,115,52,908).  visible(wk2gt_1,908). happensAt(inactive(wk2gt_1),908). orientation(wk2gt_2,142,908). coords(wk2gt_2,99,63,908).  visible(wk2gt_2,908). happensAt(inactive(wk2gt_2),908).
orientation(wk2gt_1,127,909). coords(wk2gt_1,115,51,909).  visible(wk2gt_1,909). happensAt(inactive(wk2gt_1),909). orientation(wk2gt_2,142,909). coords(wk2gt_2,99,63,909).  visible(wk2gt_2,909). happensAt(inactive(wk2gt_2),909).
orientation(wk2gt_1,127,910). coords(wk2gt_1,115,51,910).  visible(wk2gt_1,910). happensAt(inactive(wk2gt_1),910). orientation(wk2gt_2,142,910). coords(wk2gt_2,99,63,910).  visible(wk2gt_2,910). happensAt(inactive(wk2gt_2),910).
orientation(wk2gt_1,127,911). coords(wk2gt_1,115,51,911).  visible(wk2gt_1,911). happensAt(inactive(wk2gt_1),911). orientation(wk2gt_2,142,911). coords(wk2gt_2,99,63,911).  visible(wk2gt_2,911). happensAt(inactive(wk2gt_2),911).
orientation(wk2gt_1,127,912). coords(wk2gt_1,115,51,912).  visible(wk2gt_1,912). happensAt(inactive(wk2gt_1),912). orientation(wk2gt_2,142,912). coords(wk2gt_2,99,63,912).  visible(wk2gt_2,912). happensAt(inactive(wk2gt_2),912).
orientation(wk2gt_1,127,913). coords(wk2gt_1,115,51,913).  visible(wk2gt_1,913). happensAt(inactive(wk2gt_1),913). orientation(wk2gt_2,142,913). coords(wk2gt_2,99,63,913).  visible(wk2gt_2,913). happensAt(inactive(wk2gt_2),913).
orientation(wk2gt_1,127,914). coords(wk2gt_1,115,51,914).  visible(wk2gt_1,914). happensAt(inactive(wk2gt_1),914). orientation(wk2gt_2,142,914). coords(wk2gt_2,99,63,914).  visible(wk2gt_2,914). happensAt(inactive(wk2gt_2),914).
orientation(wk2gt_1,127,915). coords(wk2gt_1,115,51,915).  visible(wk2gt_1,915). happensAt(inactive(wk2gt_1),915). orientation(wk2gt_2,142,915). coords(wk2gt_2,99,63,915).  visible(wk2gt_2,915). happensAt(inactive(wk2gt_2),915).
orientation(wk2gt_1,127,916). coords(wk2gt_1,116,51,916).  visible(wk2gt_1,916). happensAt(inactive(wk2gt_1),916). orientation(wk2gt_2,142,916). coords(wk2gt_2,99,63,916).  visible(wk2gt_2,916). happensAt(inactive(wk2gt_2),916).
orientation(wk2gt_1,132,917). coords(wk2gt_1,116,51,917).  visible(wk2gt_1,917). happensAt(inactive(wk2gt_1),917). orientation(wk2gt_2,142,917). coords(wk2gt_2,99,63,917).  visible(wk2gt_2,917). happensAt(inactive(wk2gt_2),917).
orientation(wk2gt_1,132,918). coords(wk2gt_1,117,51,918).  visible(wk2gt_1,918). happensAt(inactive(wk2gt_1),918). orientation(wk2gt_2,142,918). coords(wk2gt_2,99,63,918).  visible(wk2gt_2,918). happensAt(inactive(wk2gt_2),918).
orientation(wk2gt_1,132,919). coords(wk2gt_1,117,51,919).  visible(wk2gt_1,919). happensAt(inactive(wk2gt_1),919). orientation(wk2gt_2,142,919). coords(wk2gt_2,99,63,919).  visible(wk2gt_2,919). happensAt(inactive(wk2gt_2),919).
orientation(wk2gt_1,133,920). coords(wk2gt_1,118,51,920).  visible(wk2gt_1,920). happensAt(inactive(wk2gt_1),920). orientation(wk2gt_2,142,920). coords(wk2gt_2,99,63,920).  visible(wk2gt_2,920). happensAt(inactive(wk2gt_2),920).
orientation(wk2gt_1,133,921). coords(wk2gt_1,118,51,921).  visible(wk2gt_1,921). happensAt(walking(wk2gt_1),921). orientation(wk2gt_2,142,921). coords(wk2gt_2,99,63,921).  visible(wk2gt_2,921). happensAt(inactive(wk2gt_2),921).
orientation(wk2gt_1,133,922). coords(wk2gt_1,119,50,922).  visible(wk2gt_1,922). happensAt(walking(wk2gt_1),922). orientation(wk2gt_2,142,922). coords(wk2gt_2,99,63,922).  visible(wk2gt_2,922). happensAt(inactive(wk2gt_2),922).
orientation(wk2gt_1,133,923). coords(wk2gt_1,119,50,923).  visible(wk2gt_1,923). happensAt(walking(wk2gt_1),923). orientation(wk2gt_2,142,923). coords(wk2gt_2,99,63,923).  visible(wk2gt_2,923). happensAt(inactive(wk2gt_2),923).
orientation(wk2gt_1,133,924). coords(wk2gt_1,120,50,924).  visible(wk2gt_1,924). happensAt(walking(wk2gt_1),924). orientation(wk2gt_2,142,924). coords(wk2gt_2,99,63,924).  visible(wk2gt_2,924). happensAt(inactive(wk2gt_2),924).
#show holdsAt/2.