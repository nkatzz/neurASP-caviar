

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

	person(fra1gt_6;fra1gt_7).
orientation(fra1gt_6,137,318). coords(fra1gt_6,162,78,318).  visible(fra1gt_6,318). orientation(fra1gt_7,142,318). coords(fra1gt_7,141,65,318).  visible(fra1gt_7,318).
orientation(fra1gt_6,137,319). coords(fra1gt_6,162,78,319).  visible(fra1gt_6,319). orientation(fra1gt_7,142,319). coords(fra1gt_7,142,66,319).  visible(fra1gt_7,319).
orientation(fra1gt_6,137,320). coords(fra1gt_6,162,78,320).  visible(fra1gt_6,320). orientation(fra1gt_7,142,320). coords(fra1gt_7,142,66,320).  visible(fra1gt_7,320).
orientation(fra1gt_6,137,321). coords(fra1gt_6,159,77,321).  visible(fra1gt_6,321). orientation(fra1gt_7,142,321). coords(fra1gt_7,143,66,321).  visible(fra1gt_7,321).
orientation(fra1gt_6,137,322). coords(fra1gt_6,158,77,322).  visible(fra1gt_6,322). orientation(fra1gt_7,142,322). coords(fra1gt_7,142,64,322).  visible(fra1gt_7,322).
orientation(fra1gt_6,137,323). coords(fra1gt_6,156,77,323).  visible(fra1gt_6,323). orientation(fra1gt_7,142,323). coords(fra1gt_7,143,64,323).  visible(fra1gt_7,323).
orientation(fra1gt_6,137,324). coords(fra1gt_6,156,76,324).  visible(fra1gt_6,324). orientation(fra1gt_7,142,324). coords(fra1gt_7,153,65,324).  visible(fra1gt_7,324).
orientation(fra1gt_6,137,325). coords(fra1gt_6,155,76,325).  visible(fra1gt_6,325). orientation(fra1gt_7,142,325). coords(fra1gt_7,153,65,325).  visible(fra1gt_7,325).
orientation(fra1gt_6,137,326). coords(fra1gt_6,156,76,326).  visible(fra1gt_6,326). orientation(fra1gt_7,142,326). coords(fra1gt_7,151,65,326).  visible(fra1gt_7,326).
orientation(fra1gt_6,137,327). coords(fra1gt_6,156,76,327).  visible(fra1gt_6,327). orientation(fra1gt_7,142,327). coords(fra1gt_7,151,64,327).  visible(fra1gt_7,327).
orientation(fra1gt_6,137,328). coords(fra1gt_6,155,76,328).  visible(fra1gt_6,328). orientation(fra1gt_7,142,328). coords(fra1gt_7,151,64,328).  visible(fra1gt_7,328).
orientation(fra1gt_6,137,329). coords(fra1gt_6,155,76,329).  visible(fra1gt_6,329). orientation(fra1gt_7,142,329). coords(fra1gt_7,154,65,329).  visible(fra1gt_7,329).
orientation(fra1gt_6,134,330). coords(fra1gt_6,155,76,330).  visible(fra1gt_6,330). orientation(fra1gt_7,142,330). coords(fra1gt_7,157,65,330).  visible(fra1gt_7,330).
orientation(fra1gt_6,134,331). coords(fra1gt_6,155,75,331).  visible(fra1gt_6,331). orientation(fra1gt_7,142,331). coords(fra1gt_7,158,65,331).  visible(fra1gt_7,331).
orientation(fra1gt_6,134,332). coords(fra1gt_6,156,74,332).  visible(fra1gt_6,332). orientation(fra1gt_7,142,332). coords(fra1gt_7,160,65,332).  visible(fra1gt_7,332).
orientation(fra1gt_6,134,333). coords(fra1gt_6,157,74,333).  visible(fra1gt_6,333). orientation(fra1gt_7,142,333). coords(fra1gt_7,159,67,333).  visible(fra1gt_7,333).
orientation(fra1gt_6,134,334). coords(fra1gt_6,158,76,334).  visible(fra1gt_6,334). orientation(fra1gt_7,142,334). coords(fra1gt_7,159,68,334).  visible(fra1gt_7,334).
orientation(fra1gt_6,134,335). coords(fra1gt_6,157,76,335).  visible(fra1gt_6,335). orientation(fra1gt_7,142,335). coords(fra1gt_7,161,68,335).  visible(fra1gt_7,335).
orientation(fra1gt_6,134,336). coords(fra1gt_6,157,76,336).  visible(fra1gt_6,336). orientation(fra1gt_7,142,336). coords(fra1gt_7,163,69,336).  visible(fra1gt_7,336).
orientation(fra1gt_6,134,337). coords(fra1gt_6,157,76,337).  visible(fra1gt_6,337). orientation(fra1gt_7,142,337). coords(fra1gt_7,165,69,337).  visible(fra1gt_7,337).
orientation(fra1gt_6,134,338). coords(fra1gt_6,157,77,338).  visible(fra1gt_6,338). orientation(fra1gt_7,142,338). coords(fra1gt_7,165,69,338).  visible(fra1gt_7,338).
orientation(fra1gt_6,134,339). coords(fra1gt_6,158,77,339).  visible(fra1gt_6,339). orientation(fra1gt_7,142,339). coords(fra1gt_7,165,69,339).  visible(fra1gt_7,339).
orientation(fra1gt_6,134,340). coords(fra1gt_6,158,77,340).  visible(fra1gt_6,340). orientation(fra1gt_7,142,340). coords(fra1gt_7,167,70,340).  visible(fra1gt_7,340).
orientation(fra1gt_6,134,341). coords(fra1gt_6,158,77,341).  visible(fra1gt_6,341). orientation(fra1gt_7,142,341). coords(fra1gt_7,168,70,341).  visible(fra1gt_7,341).
orientation(fra1gt_6,134,342). coords(fra1gt_6,160,77,342).  visible(fra1gt_6,342). orientation(fra1gt_7,142,342). coords(fra1gt_7,167,70,342).  visible(fra1gt_7,342).
orientation(fra1gt_6,134,343). coords(fra1gt_6,160,77,343).  visible(fra1gt_6,343). orientation(fra1gt_7,142,343). coords(fra1gt_7,167,70,343).  visible(fra1gt_7,343).
happensAt(disappear(fra1gt_6),342). happensAt(disappear(fra1gt_7),342).
1{simpleEvent(T,fra1gt_6,active); simpleEvent(T,fra1gt_6,inactive); simpleEvent(T,fra1gt_6,walking); simpleEvent(T,fra1gt_6,running)}1 :- T=318..342.
1{simpleEvent(T,fra1gt_7,active); simpleEvent(T,fra1gt_7,inactive); simpleEvent(T,fra1gt_7,walking); simpleEvent(T,fra1gt_7,running)}1 :- T=318..342.
:- holdsAt(moving(fra1gt_6,fra1gt_7),318). :- holdsAt(moving(fra1gt_7,fra1gt_6),318).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),319). :- not holdsAt(moving(fra1gt_7,fra1gt_6),319).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),320). :- not holdsAt(moving(fra1gt_7,fra1gt_6),320).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),321). :- not holdsAt(moving(fra1gt_7,fra1gt_6),321).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),322). :- not holdsAt(moving(fra1gt_7,fra1gt_6),322).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),323). :- not holdsAt(moving(fra1gt_7,fra1gt_6),323).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),324). :- not holdsAt(moving(fra1gt_7,fra1gt_6),324).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),325). :- not holdsAt(moving(fra1gt_7,fra1gt_6),325).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),326). :- not holdsAt(moving(fra1gt_7,fra1gt_6),326).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),327). :- not holdsAt(moving(fra1gt_7,fra1gt_6),327).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),328). :- not holdsAt(moving(fra1gt_7,fra1gt_6),328).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),329). :- not holdsAt(moving(fra1gt_7,fra1gt_6),329).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),330). :- not holdsAt(moving(fra1gt_7,fra1gt_6),330).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),331). :- not holdsAt(moving(fra1gt_7,fra1gt_6),331).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),332). :- not holdsAt(moving(fra1gt_7,fra1gt_6),332).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),333). :- not holdsAt(moving(fra1gt_7,fra1gt_6),333).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),334). :- not holdsAt(moving(fra1gt_7,fra1gt_6),334).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),335). :- not holdsAt(moving(fra1gt_7,fra1gt_6),335).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),336). :- not holdsAt(moving(fra1gt_7,fra1gt_6),336).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),337). :- not holdsAt(moving(fra1gt_7,fra1gt_6),337).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),338). :- not holdsAt(moving(fra1gt_7,fra1gt_6),338).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),339). :- not holdsAt(moving(fra1gt_7,fra1gt_6),339).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),340). :- not holdsAt(moving(fra1gt_7,fra1gt_6),340).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),341). :- not holdsAt(moving(fra1gt_7,fra1gt_6),341).
:- not holdsAt(moving(fra1gt_6,fra1gt_7),342). :- not holdsAt(moving(fra1gt_7,fra1gt_6),342).
:- holdsAt(moving(fra1gt_6,fra1gt_7),343). :- holdsAt(moving(fra1gt_7,fra1gt_6),343).
:- simpleEvent(T,P,inactive), T>318 , T<342.
:- simpleEvent(T,P,running), T>318 , T<342.

#show simpleEvent/3.