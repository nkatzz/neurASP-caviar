

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
orientation(wk2gt_1,127,888). coords(wk2gt_1,115,53,888).  visible(wk2gt_1,888). orientation(wk2gt_2,142,888). coords(wk2gt_2,99,63,888).  visible(wk2gt_2,888).
orientation(wk2gt_1,127,889). coords(wk2gt_1,115,53,889).  visible(wk2gt_1,889). orientation(wk2gt_2,142,889). coords(wk2gt_2,99,63,889).  visible(wk2gt_2,889).
orientation(wk2gt_1,127,890). coords(wk2gt_1,115,53,890).  visible(wk2gt_1,890). orientation(wk2gt_2,142,890). coords(wk2gt_2,99,63,890).  visible(wk2gt_2,890).
orientation(wk2gt_1,127,891). coords(wk2gt_1,115,53,891).  visible(wk2gt_1,891). orientation(wk2gt_2,142,891). coords(wk2gt_2,99,63,891).  visible(wk2gt_2,891).
orientation(wk2gt_1,127,892). coords(wk2gt_1,115,53,892).  visible(wk2gt_1,892). orientation(wk2gt_2,142,892). coords(wk2gt_2,99,63,892).  visible(wk2gt_2,892).
orientation(wk2gt_1,127,893). coords(wk2gt_1,115,53,893).  visible(wk2gt_1,893). orientation(wk2gt_2,142,893). coords(wk2gt_2,99,63,893).  visible(wk2gt_2,893).
orientation(wk2gt_1,127,894). coords(wk2gt_1,115,53,894).  visible(wk2gt_1,894). orientation(wk2gt_2,142,894). coords(wk2gt_2,99,63,894).  visible(wk2gt_2,894).
orientation(wk2gt_1,127,895). coords(wk2gt_1,115,53,895).  visible(wk2gt_1,895). orientation(wk2gt_2,142,895). coords(wk2gt_2,99,63,895).  visible(wk2gt_2,895).
orientation(wk2gt_1,127,896). coords(wk2gt_1,115,53,896).  visible(wk2gt_1,896). orientation(wk2gt_2,142,896). coords(wk2gt_2,99,63,896).  visible(wk2gt_2,896).
orientation(wk2gt_1,127,897). coords(wk2gt_1,115,53,897).  visible(wk2gt_1,897). orientation(wk2gt_2,142,897). coords(wk2gt_2,99,63,897).  visible(wk2gt_2,897).
orientation(wk2gt_1,127,898). coords(wk2gt_1,115,53,898).  visible(wk2gt_1,898). orientation(wk2gt_2,142,898). coords(wk2gt_2,99,63,898).  visible(wk2gt_2,898).
orientation(wk2gt_1,127,899). coords(wk2gt_1,115,53,899).  visible(wk2gt_1,899). orientation(wk2gt_2,142,899). coords(wk2gt_2,99,63,899).  visible(wk2gt_2,899).
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
happensAt(disappear(wk2gt_1),912). happensAt(disappear(wk2gt_2),912).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=888..912.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=888..912.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),888). :- holdsAt(meeting(wk2gt_2,wk2gt_1),888).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),889). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),889).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),890). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),890).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),891). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),891).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),892). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),892).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),893). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),893).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),894). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),894).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),895). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),895).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),896). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),896).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),897). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),897).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),898). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),898).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),899). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),899).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),900). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),900).
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
:- holdsAt(meeting(wk2gt_1,wk2gt_2),913). :- holdsAt(meeting(wk2gt_2,wk2gt_1),913).
:- simpleEvent(T,P,walking), T>888 , T<912.
:- simpleEvent(T,P,running), T>888 , T<912.

#show simpleEvent/3.