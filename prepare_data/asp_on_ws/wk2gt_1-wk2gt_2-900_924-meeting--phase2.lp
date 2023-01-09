

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,127,900). coords(wk2gt_1,115,53,900).  visible(wk2gt_1,900). orientation(wk2gt_2,142,900). coords(wk2gt_2,99,63,900).  visible(wk2gt_2,900).
orientation(wk2gt_1,127,901). coords(wk2gt_1,115,53,901).  visible(wk2gt_1,901). orientation(wk2gt_2,142,901). coords(wk2gt_2,99,63,901).  visible(wk2gt_2,901).
orientation(wk2gt_1,127,902). coords(wk2gt_1,115,53,902).  visible(wk2gt_1,902). orientation(wk2gt_2,142,902). coords(wk2gt_2,99,63,902).  visible(wk2gt_2,902).
orientation(wk2gt_1,127,903). coords(wk2gt_1,115,53,903).  visible(wk2gt_1,903). orientation(wk2gt_2,142,903). coords(wk2gt_2,99,63,903).  visible(wk2gt_2,903).
orientation(wk2gt_1,127,904). coords(wk2gt_1,115,53,904).  visible(wk2gt_1,904). orientation(wk2gt_2,142,904). coords(wk2gt_2,99,63,904).  visible(wk2gt_2,904).
orientation(wk2gt_1,127,905). coords(wk2gt_1,115,53,905).  visible(wk2gt_1,905). orientation(wk2gt_2,142,905). coords(wk2gt_2,99,63,905).  visible(wk2gt_2,905).
orientation(wk2gt_1,127,906). coords(wk2gt_1,115,53,906).  visible(wk2gt_1,906). orientation(wk2gt_2,142,906). coords(wk2gt_2,99,63,906).  visible(wk2gt_2,906).
orientation(wk2gt_1,127,907). coords(wk2gt_1,115,53,907).  visible(wk2gt_1,907). orientation(wk2gt_2,142,907). coords(wk2gt_2,99,63,907).  visible(wk2gt_2,907).
orientation(wk2gt_1,124,908). coords(wk2gt_1,115,52,908).  visible(wk2gt_1,908). orientation(wk2gt_2,142,908). coords(wk2gt_2,99,63,908).  visible(wk2gt_2,908).
orientation(wk2gt_1,127,909). coords(wk2gt_1,115,51,909).  visible(wk2gt_1,909). orientation(wk2gt_2,142,909). coords(wk2gt_2,99,63,909).  visible(wk2gt_2,909).
orientation(wk2gt_1,127,910). coords(wk2gt_1,115,51,910).  visible(wk2gt_1,910). orientation(wk2gt_2,142,910). coords(wk2gt_2,99,63,910).  visible(wk2gt_2,910).
orientation(wk2gt_1,127,911). coords(wk2gt_1,115,51,911).  visible(wk2gt_1,911). orientation(wk2gt_2,142,911). coords(wk2gt_2,99,63,911).  visible(wk2gt_2,911).
orientation(wk2gt_1,127,912). coords(wk2gt_1,115,51,912).  visible(wk2gt_1,912). orientation(wk2gt_2,142,912). coords(wk2gt_2,99,63,912).  visible(wk2gt_2,912).
orientation(wk2gt_1,127,913). coords(wk2gt_1,115,51,913).  visible(wk2gt_1,913). orientation(wk2gt_2,142,913). coords(wk2gt_2,99,63,913).  visible(wk2gt_2,913).
orientation(wk2gt_1,127,914). coords(wk2gt_1,115,51,914).  visible(wk2gt_1,914). orientation(wk2gt_2,142,914). coords(wk2gt_2,99,63,914).  visible(wk2gt_2,914).
orientation(wk2gt_1,127,915). coords(wk2gt_1,115,51,915).  visible(wk2gt_1,915). orientation(wk2gt_2,142,915). coords(wk2gt_2,99,63,915).  visible(wk2gt_2,915).
orientation(wk2gt_1,127,916). coords(wk2gt_1,116,51,916).  visible(wk2gt_1,916). orientation(wk2gt_2,142,916). coords(wk2gt_2,99,63,916).  visible(wk2gt_2,916).
orientation(wk2gt_1,132,917). coords(wk2gt_1,116,51,917).  visible(wk2gt_1,917). orientation(wk2gt_2,142,917). coords(wk2gt_2,99,63,917).  visible(wk2gt_2,917).
orientation(wk2gt_1,132,918). coords(wk2gt_1,117,51,918).  visible(wk2gt_1,918). orientation(wk2gt_2,142,918). coords(wk2gt_2,99,63,918).  visible(wk2gt_2,918).
orientation(wk2gt_1,132,919). coords(wk2gt_1,117,51,919).  visible(wk2gt_1,919). orientation(wk2gt_2,142,919). coords(wk2gt_2,99,63,919).  visible(wk2gt_2,919).
orientation(wk2gt_1,133,920). coords(wk2gt_1,118,51,920).  visible(wk2gt_1,920). orientation(wk2gt_2,142,920). coords(wk2gt_2,99,63,920).  visible(wk2gt_2,920).
orientation(wk2gt_1,133,921). coords(wk2gt_1,118,51,921).  visible(wk2gt_1,921). orientation(wk2gt_2,142,921). coords(wk2gt_2,99,63,921).  visible(wk2gt_2,921).
orientation(wk2gt_1,133,922). coords(wk2gt_1,119,50,922).  visible(wk2gt_1,922). orientation(wk2gt_2,142,922). coords(wk2gt_2,99,63,922).  visible(wk2gt_2,922).
orientation(wk2gt_1,133,923). coords(wk2gt_1,119,50,923).  visible(wk2gt_1,923). orientation(wk2gt_2,142,923). coords(wk2gt_2,99,63,923).  visible(wk2gt_2,923).
orientation(wk2gt_1,133,924). coords(wk2gt_1,120,50,924).  visible(wk2gt_1,924). orientation(wk2gt_2,142,924). coords(wk2gt_2,99,63,924).  visible(wk2gt_2,924).
orientation(wk2gt_1,133,925). coords(wk2gt_1,120,50,925).  visible(wk2gt_1,925). orientation(wk2gt_2,142,925). coords(wk2gt_2,99,63,925).  visible(wk2gt_2,925).
happensAt(disappear(wk2gt_1),924). happensAt(disappear(wk2gt_2),924).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=900..924.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=900..924.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),900). :- holdsAt(meeting(wk2gt_2,wk2gt_1),900).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),901). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),901).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),902). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),902).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),903). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),903).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),904). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),904).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),905). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),905).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),906). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),906).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),907). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),907).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),908). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),908).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),909). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),909).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),910). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),910).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),911). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),911).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),912). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),912).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),913). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),913).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),914). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),914).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),915). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),915).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),916). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),916).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),917). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),917).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),918). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),918).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),919). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),919).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),920). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),920).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),921). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),921).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),922). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),922).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),923). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),923).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),924). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),924).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),925). :- holdsAt(meeting(wk2gt_2,wk2gt_1),925).
:- simpleEvent(T,P,walking), T>900 , T<924.
:- simpleEvent(T,P,running), T>900 , T<924.

#show simpleEvent/3.