

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
orientation(wk2gt_1,125,312). coords(wk2gt_1,115,53,312).  visible(wk2gt_1,312). orientation(wk2gt_2,141,312). coords(wk2gt_2,98,63,312).  visible(wk2gt_2,312).
orientation(wk2gt_1,125,313). coords(wk2gt_1,115,53,313).  visible(wk2gt_1,313). orientation(wk2gt_2,141,313). coords(wk2gt_2,98,63,313).  visible(wk2gt_2,313).
orientation(wk2gt_1,125,314). coords(wk2gt_1,115,53,314).  visible(wk2gt_1,314). orientation(wk2gt_2,141,314). coords(wk2gt_2,98,63,314).  visible(wk2gt_2,314).
orientation(wk2gt_1,125,315). coords(wk2gt_1,115,53,315).  visible(wk2gt_1,315). orientation(wk2gt_2,141,315). coords(wk2gt_2,98,63,315).  visible(wk2gt_2,315).
orientation(wk2gt_1,125,316). coords(wk2gt_1,115,53,316).  visible(wk2gt_1,316). orientation(wk2gt_2,141,316). coords(wk2gt_2,98,63,316).  visible(wk2gt_2,316).
orientation(wk2gt_1,125,317). coords(wk2gt_1,115,53,317).  visible(wk2gt_1,317). orientation(wk2gt_2,141,317). coords(wk2gt_2,98,63,317).  visible(wk2gt_2,317).
orientation(wk2gt_1,125,318). coords(wk2gt_1,115,53,318).  visible(wk2gt_1,318). orientation(wk2gt_2,141,318). coords(wk2gt_2,98,63,318).  visible(wk2gt_2,318).
orientation(wk2gt_1,125,319). coords(wk2gt_1,115,53,319).  visible(wk2gt_1,319). orientation(wk2gt_2,141,319). coords(wk2gt_2,98,63,319).  visible(wk2gt_2,319).
orientation(wk2gt_1,125,320). coords(wk2gt_1,115,53,320).  visible(wk2gt_1,320). orientation(wk2gt_2,141,320). coords(wk2gt_2,98,63,320).  visible(wk2gt_2,320).
orientation(wk2gt_1,125,321). coords(wk2gt_1,115,53,321).  visible(wk2gt_1,321). orientation(wk2gt_2,141,321). coords(wk2gt_2,98,63,321).  visible(wk2gt_2,321).
orientation(wk2gt_1,125,322). coords(wk2gt_1,115,53,322).  visible(wk2gt_1,322). orientation(wk2gt_2,141,322). coords(wk2gt_2,98,63,322).  visible(wk2gt_2,322).
orientation(wk2gt_1,125,323). coords(wk2gt_1,115,53,323).  visible(wk2gt_1,323). orientation(wk2gt_2,141,323). coords(wk2gt_2,98,63,323).  visible(wk2gt_2,323).
orientation(wk2gt_1,125,324). coords(wk2gt_1,115,53,324).  visible(wk2gt_1,324). orientation(wk2gt_2,141,324). coords(wk2gt_2,98,63,324).  visible(wk2gt_2,324).
orientation(wk2gt_1,125,325). coords(wk2gt_1,115,53,325).  visible(wk2gt_1,325). orientation(wk2gt_2,141,325). coords(wk2gt_2,98,63,325).  visible(wk2gt_2,325).
orientation(wk2gt_1,125,326). coords(wk2gt_1,115,53,326).  visible(wk2gt_1,326). orientation(wk2gt_2,141,326). coords(wk2gt_2,98,63,326).  visible(wk2gt_2,326).
orientation(wk2gt_1,125,327). coords(wk2gt_1,115,53,327).  visible(wk2gt_1,327). orientation(wk2gt_2,141,327). coords(wk2gt_2,98,63,327).  visible(wk2gt_2,327).
orientation(wk2gt_1,125,328). coords(wk2gt_1,115,53,328).  visible(wk2gt_1,328). orientation(wk2gt_2,141,328). coords(wk2gt_2,98,63,328).  visible(wk2gt_2,328).
orientation(wk2gt_1,125,329). coords(wk2gt_1,115,53,329).  visible(wk2gt_1,329). orientation(wk2gt_2,141,329). coords(wk2gt_2,98,63,329).  visible(wk2gt_2,329).
orientation(wk2gt_1,125,330). coords(wk2gt_1,115,53,330).  visible(wk2gt_1,330). orientation(wk2gt_2,141,330). coords(wk2gt_2,98,63,330).  visible(wk2gt_2,330).
orientation(wk2gt_1,125,331). coords(wk2gt_1,115,53,331).  visible(wk2gt_1,331). orientation(wk2gt_2,141,331). coords(wk2gt_2,98,63,331).  visible(wk2gt_2,331).
orientation(wk2gt_1,125,332). coords(wk2gt_1,115,53,332).  visible(wk2gt_1,332). orientation(wk2gt_2,141,332). coords(wk2gt_2,98,63,332).  visible(wk2gt_2,332).
orientation(wk2gt_1,125,333). coords(wk2gt_1,115,53,333).  visible(wk2gt_1,333). orientation(wk2gt_2,141,333). coords(wk2gt_2,98,63,333).  visible(wk2gt_2,333).
orientation(wk2gt_1,125,334). coords(wk2gt_1,115,53,334).  visible(wk2gt_1,334). orientation(wk2gt_2,141,334). coords(wk2gt_2,98,63,334).  visible(wk2gt_2,334).
orientation(wk2gt_1,125,335). coords(wk2gt_1,115,53,335).  visible(wk2gt_1,335). orientation(wk2gt_2,141,335). coords(wk2gt_2,98,63,335).  visible(wk2gt_2,335).
orientation(wk2gt_1,125,336). coords(wk2gt_1,115,53,336).  visible(wk2gt_1,336). orientation(wk2gt_2,141,336). coords(wk2gt_2,98,63,336).  visible(wk2gt_2,336).
orientation(wk2gt_1,125,337). coords(wk2gt_1,115,53,337).  visible(wk2gt_1,337). orientation(wk2gt_2,141,337). coords(wk2gt_2,98,63,337).  visible(wk2gt_2,337).
happensAt(disappear(wk2gt_1),336). happensAt(disappear(wk2gt_2),336).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=312..336.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=312..336.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),312). :- holdsAt(meeting(wk2gt_2,wk2gt_1),312).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),313). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),313).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),314). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),314).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),315). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),315).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),316). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),316).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),317). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),317).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),318). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),318).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),319). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),319).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),320). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),320).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),321). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),321).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),322). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),322).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),323). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),323).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),324). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),324).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),325). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),325).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),326). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),326).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),327). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),327).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),328). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),328).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),329). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),329).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),330). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),330).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),331). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),331).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),332). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),332).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),333). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),333).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),334). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),334).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),335). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),335).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),336). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),336).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),337). :- holdsAt(meeting(wk2gt_2,wk2gt_1),337).
:- simpleEvent(T,P,walking), T>312 , T<336.
:- simpleEvent(T,P,running), T>312 , T<336.

#show simpleEvent/3.