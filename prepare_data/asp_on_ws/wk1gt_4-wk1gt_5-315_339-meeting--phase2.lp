

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,159,315). coords(wk1gt_4,52,176,315).  visible(wk1gt_4,315). orientation(wk1gt_5,162,315). coords(wk1gt_5,50,163,315).  visible(wk1gt_5,315).
orientation(wk1gt_4,157,316). coords(wk1gt_4,52,176,316).  visible(wk1gt_4,316). orientation(wk1gt_5,162,316). coords(wk1gt_5,50,163,316).  visible(wk1gt_5,316).
orientation(wk1gt_4,160,317). coords(wk1gt_4,52,176,317).  visible(wk1gt_4,317). orientation(wk1gt_5,162,317). coords(wk1gt_5,50,163,317).  visible(wk1gt_5,317).
orientation(wk1gt_4,160,318). coords(wk1gt_4,52,176,318).  visible(wk1gt_4,318). orientation(wk1gt_5,162,318). coords(wk1gt_5,50,163,318).  visible(wk1gt_5,318).
orientation(wk1gt_4,160,319). coords(wk1gt_4,52,176,319).  visible(wk1gt_4,319). orientation(wk1gt_5,162,319). coords(wk1gt_5,50,163,319).  visible(wk1gt_5,319).
orientation(wk1gt_4,160,320). coords(wk1gt_4,52,176,320).  visible(wk1gt_4,320). orientation(wk1gt_5,162,320). coords(wk1gt_5,51,163,320).  visible(wk1gt_5,320).
orientation(wk1gt_4,161,321). coords(wk1gt_4,52,176,321).  visible(wk1gt_4,321). orientation(wk1gt_5,162,321). coords(wk1gt_5,50,163,321).  visible(wk1gt_5,321).
orientation(wk1gt_4,162,322). coords(wk1gt_4,52,176,322).  visible(wk1gt_4,322). orientation(wk1gt_5,162,322). coords(wk1gt_5,50,163,322).  visible(wk1gt_5,322).
orientation(wk1gt_4,161,323). coords(wk1gt_4,52,176,323).  visible(wk1gt_4,323). orientation(wk1gt_5,162,323). coords(wk1gt_5,50,163,323).  visible(wk1gt_5,323).
orientation(wk1gt_4,161,324). coords(wk1gt_4,52,176,324).  visible(wk1gt_4,324). orientation(wk1gt_5,162,324). coords(wk1gt_5,50,163,324).  visible(wk1gt_5,324).
orientation(wk1gt_4,158,325). coords(wk1gt_4,52,176,325).  visible(wk1gt_4,325). orientation(wk1gt_5,162,325). coords(wk1gt_5,50,163,325).  visible(wk1gt_5,325).
orientation(wk1gt_4,166,326). coords(wk1gt_4,52,176,326).  visible(wk1gt_4,326). orientation(wk1gt_5,162,326). coords(wk1gt_5,50,163,326).  visible(wk1gt_5,326).
orientation(wk1gt_4,162,327). coords(wk1gt_4,52,176,327).  visible(wk1gt_4,327). orientation(wk1gt_5,162,327). coords(wk1gt_5,50,163,327).  visible(wk1gt_5,327).
orientation(wk1gt_4,162,328). coords(wk1gt_4,52,176,328).  visible(wk1gt_4,328). orientation(wk1gt_5,162,328). coords(wk1gt_5,51,163,328).  visible(wk1gt_5,328).
orientation(wk1gt_4,158,329). coords(wk1gt_4,52,176,329).  visible(wk1gt_4,329). orientation(wk1gt_5,162,329). coords(wk1gt_5,51,163,329).  visible(wk1gt_5,329).
orientation(wk1gt_4,160,330). coords(wk1gt_4,52,176,330).  visible(wk1gt_4,330). orientation(wk1gt_5,162,330). coords(wk1gt_5,51,163,330).  visible(wk1gt_5,330).
orientation(wk1gt_4,157,331). coords(wk1gt_4,52,176,331).  visible(wk1gt_4,331). orientation(wk1gt_5,162,331). coords(wk1gt_5,51,163,331).  visible(wk1gt_5,331).
orientation(wk1gt_4,160,332). coords(wk1gt_4,52,176,332).  visible(wk1gt_4,332). orientation(wk1gt_5,162,332). coords(wk1gt_5,50,163,332).  visible(wk1gt_5,332).
orientation(wk1gt_4,157,333). coords(wk1gt_4,52,176,333).  visible(wk1gt_4,333). orientation(wk1gt_5,162,333). coords(wk1gt_5,50,163,333).  visible(wk1gt_5,333).
orientation(wk1gt_4,161,334). coords(wk1gt_4,52,176,334).  visible(wk1gt_4,334). orientation(wk1gt_5,162,334). coords(wk1gt_5,50,163,334).  visible(wk1gt_5,334).
orientation(wk1gt_4,165,335). coords(wk1gt_4,52,176,335).  visible(wk1gt_4,335). orientation(wk1gt_5,162,335). coords(wk1gt_5,50,163,335).  visible(wk1gt_5,335).
orientation(wk1gt_4,164,336). coords(wk1gt_4,52,176,336).  visible(wk1gt_4,336). orientation(wk1gt_5,162,336). coords(wk1gt_5,50,163,336).  visible(wk1gt_5,336).
orientation(wk1gt_4,156,337). coords(wk1gt_4,52,176,337).  visible(wk1gt_4,337). orientation(wk1gt_5,162,337). coords(wk1gt_5,50,163,337).  visible(wk1gt_5,337).
orientation(wk1gt_4,163,338). coords(wk1gt_4,52,176,338).  visible(wk1gt_4,338). orientation(wk1gt_5,162,338). coords(wk1gt_5,50,163,338).  visible(wk1gt_5,338).
orientation(wk1gt_4,160,339). coords(wk1gt_4,52,176,339).  visible(wk1gt_4,339). orientation(wk1gt_5,162,339). coords(wk1gt_5,50,163,339).  visible(wk1gt_5,339).
orientation(wk1gt_4,160,340). coords(wk1gt_4,52,176,340).  visible(wk1gt_4,340). orientation(wk1gt_5,162,340). coords(wk1gt_5,50,163,340).  visible(wk1gt_5,340).
happensAt(disappear(wk1gt_4),339). happensAt(disappear(wk1gt_5),339).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=315..339.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=315..339.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),315). :- holdsAt(meeting(wk1gt_5,wk1gt_4),315).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),316). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),316).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),317). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),317).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),318). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),318).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),319). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),319).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),320). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),320).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),321). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),321).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),322). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),322).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),323). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),323).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),324). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),324).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),325). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),325).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),326). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),326).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),327). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),327).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),328). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),328).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),329). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),329).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),330). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),330).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),331). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),331).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),332). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),332).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),333). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),333).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),334). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),334).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),335). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),335).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),336). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),336).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),337). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),337).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),338). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),338).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),339). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),339).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),340). :- holdsAt(meeting(wk1gt_5,wk1gt_4),340).
:- simpleEvent(T,P,walking), T>315 , T<339.
:- simpleEvent(T,P,running), T>315 , T<339.

#show simpleEvent/3.