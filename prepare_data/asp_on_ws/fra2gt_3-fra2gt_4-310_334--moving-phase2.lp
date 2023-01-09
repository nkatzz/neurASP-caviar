

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

	person(fra2gt_3;fra2gt_4).
orientation(fra2gt_3,150,310). coords(fra2gt_3,95,117,310).  visible(fra2gt_3,310). orientation(fra2gt_4,155,310). coords(fra2gt_4,97,119,310).  visible(fra2gt_4,310).
orientation(fra2gt_3,150,311). coords(fra2gt_3,95,117,311).  visible(fra2gt_3,311). orientation(fra2gt_4,155,311). coords(fra2gt_4,99,119,311).  visible(fra2gt_4,311).
orientation(fra2gt_3,150,312). coords(fra2gt_3,95,116,312).  visible(fra2gt_3,312). orientation(fra2gt_4,155,312). coords(fra2gt_4,99,119,312).  visible(fra2gt_4,312).
orientation(fra2gt_3,150,313). coords(fra2gt_3,95,116,313).  visible(fra2gt_3,313). orientation(fra2gt_4,155,313). coords(fra2gt_4,99,119,313).  visible(fra2gt_4,313).
orientation(fra2gt_3,150,314). coords(fra2gt_3,95,116,314).  visible(fra2gt_3,314). orientation(fra2gt_4,155,314). coords(fra2gt_4,99,119,314).  visible(fra2gt_4,314).
orientation(fra2gt_3,150,315). coords(fra2gt_3,96,116,315).  visible(fra2gt_3,315). orientation(fra2gt_4,155,315). coords(fra2gt_4,100,119,315).  visible(fra2gt_4,315).
orientation(fra2gt_3,150,316). coords(fra2gt_3,96,116,316).  visible(fra2gt_3,316). orientation(fra2gt_4,155,316). coords(fra2gt_4,100,119,316).  visible(fra2gt_4,316).
orientation(fra2gt_3,150,317). coords(fra2gt_3,96,116,317).  visible(fra2gt_3,317). orientation(fra2gt_4,155,317). coords(fra2gt_4,99,120,317).  visible(fra2gt_4,317).
orientation(fra2gt_3,150,318). coords(fra2gt_3,95,116,318).  visible(fra2gt_3,318). orientation(fra2gt_4,155,318). coords(fra2gt_4,97,120,318).  visible(fra2gt_4,318).
orientation(fra2gt_3,150,319). coords(fra2gt_3,94,116,319).  visible(fra2gt_3,319). orientation(fra2gt_4,155,319). coords(fra2gt_4,97,120,319).  visible(fra2gt_4,319).
orientation(fra2gt_3,150,320). coords(fra2gt_3,94,116,320).  visible(fra2gt_3,320). orientation(fra2gt_4,155,320). coords(fra2gt_4,98,121,320).  visible(fra2gt_4,320).
orientation(fra2gt_3,150,321). coords(fra2gt_3,94,116,321).  visible(fra2gt_3,321). orientation(fra2gt_4,155,321). coords(fra2gt_4,99,121,321).  visible(fra2gt_4,321).
orientation(fra2gt_3,150,322). coords(fra2gt_3,94,116,322).  visible(fra2gt_3,322). orientation(fra2gt_4,155,322). coords(fra2gt_4,101,121,322).  visible(fra2gt_4,322).
orientation(fra2gt_3,143,323). coords(fra2gt_3,92,116,323).  visible(fra2gt_3,323). orientation(fra2gt_4,155,323). coords(fra2gt_4,101,121,323).  visible(fra2gt_4,323).
orientation(fra2gt_3,143,324). coords(fra2gt_3,91,110,324).  visible(fra2gt_3,324). orientation(fra2gt_4,155,324). coords(fra2gt_4,101,122,324).  visible(fra2gt_4,324).
orientation(fra2gt_3,143,325). coords(fra2gt_3,91,110,325).  visible(fra2gt_3,325). orientation(fra2gt_4,155,325). coords(fra2gt_4,101,122,325).  visible(fra2gt_4,325).
orientation(fra2gt_3,143,326). coords(fra2gt_3,91,110,326).  visible(fra2gt_3,326). orientation(fra2gt_4,155,326). coords(fra2gt_4,101,122,326).  visible(fra2gt_4,326).
orientation(fra2gt_3,143,327). coords(fra2gt_3,90,110,327).  visible(fra2gt_3,327). orientation(fra2gt_4,155,327). coords(fra2gt_4,101,122,327).  visible(fra2gt_4,327).
orientation(fra2gt_3,143,328). coords(fra2gt_3,90,114,328).  visible(fra2gt_3,328). orientation(fra2gt_4,155,328). coords(fra2gt_4,101,123,328).  visible(fra2gt_4,328).
orientation(fra2gt_3,143,329). coords(fra2gt_3,89,114,329).  visible(fra2gt_3,329). orientation(fra2gt_4,155,329). coords(fra2gt_4,101,122,329).  visible(fra2gt_4,329).
orientation(fra2gt_3,143,330). coords(fra2gt_3,88,114,330).  visible(fra2gt_3,330). orientation(fra2gt_4,155,330). coords(fra2gt_4,102,120,330).  visible(fra2gt_4,330).
orientation(fra2gt_3,143,331). coords(fra2gt_3,88,114,331).  visible(fra2gt_3,331). orientation(fra2gt_4,155,331). coords(fra2gt_4,102,119,331).  visible(fra2gt_4,331).
orientation(fra2gt_3,138,332). coords(fra2gt_3,89,114,332).  visible(fra2gt_3,332). orientation(fra2gt_4,155,332). coords(fra2gt_4,101,117,332).  visible(fra2gt_4,332).
orientation(fra2gt_3,138,333). coords(fra2gt_3,89,114,333).  visible(fra2gt_3,333). orientation(fra2gt_4,155,333). coords(fra2gt_4,100,116,333).  visible(fra2gt_4,333).
orientation(fra2gt_3,138,334). coords(fra2gt_3,89,114,334).  visible(fra2gt_3,334). orientation(fra2gt_4,155,334). coords(fra2gt_4,99,116,334).  visible(fra2gt_4,334).
orientation(fra2gt_3,138,335). coords(fra2gt_3,89,114,335).  visible(fra2gt_3,335). orientation(fra2gt_4,155,335). coords(fra2gt_4,99,116,335).  visible(fra2gt_4,335).
happensAt(disappear(fra2gt_3),334). happensAt(disappear(fra2gt_4),334).
1{simpleEvent(T,fra2gt_3,active); simpleEvent(T,fra2gt_3,inactive); simpleEvent(T,fra2gt_3,walking); simpleEvent(T,fra2gt_3,running)}1 :- T=310..334.
1{simpleEvent(T,fra2gt_4,active); simpleEvent(T,fra2gt_4,inactive); simpleEvent(T,fra2gt_4,walking); simpleEvent(T,fra2gt_4,running)}1 :- T=310..334.
:- holdsAt(moving(fra2gt_3,fra2gt_4),310). :- holdsAt(moving(fra2gt_4,fra2gt_3),310).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),311). :- not holdsAt(moving(fra2gt_4,fra2gt_3),311).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),312). :- not holdsAt(moving(fra2gt_4,fra2gt_3),312).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),313). :- not holdsAt(moving(fra2gt_4,fra2gt_3),313).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),314). :- not holdsAt(moving(fra2gt_4,fra2gt_3),314).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),315). :- not holdsAt(moving(fra2gt_4,fra2gt_3),315).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),316). :- not holdsAt(moving(fra2gt_4,fra2gt_3),316).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),317). :- not holdsAt(moving(fra2gt_4,fra2gt_3),317).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),318). :- not holdsAt(moving(fra2gt_4,fra2gt_3),318).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),319). :- not holdsAt(moving(fra2gt_4,fra2gt_3),319).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),320). :- not holdsAt(moving(fra2gt_4,fra2gt_3),320).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),321). :- not holdsAt(moving(fra2gt_4,fra2gt_3),321).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),322). :- not holdsAt(moving(fra2gt_4,fra2gt_3),322).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),323). :- not holdsAt(moving(fra2gt_4,fra2gt_3),323).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),324). :- not holdsAt(moving(fra2gt_4,fra2gt_3),324).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),325). :- not holdsAt(moving(fra2gt_4,fra2gt_3),325).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),326). :- not holdsAt(moving(fra2gt_4,fra2gt_3),326).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),327). :- not holdsAt(moving(fra2gt_4,fra2gt_3),327).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),328). :- not holdsAt(moving(fra2gt_4,fra2gt_3),328).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),329). :- not holdsAt(moving(fra2gt_4,fra2gt_3),329).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),330). :- not holdsAt(moving(fra2gt_4,fra2gt_3),330).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),331). :- not holdsAt(moving(fra2gt_4,fra2gt_3),331).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),332). :- not holdsAt(moving(fra2gt_4,fra2gt_3),332).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),333). :- not holdsAt(moving(fra2gt_4,fra2gt_3),333).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),334). :- not holdsAt(moving(fra2gt_4,fra2gt_3),334).
:- holdsAt(moving(fra2gt_3,fra2gt_4),335). :- holdsAt(moving(fra2gt_4,fra2gt_3),335).
:- simpleEvent(T,P,inactive), T>310 , T<334.
:- simpleEvent(T,P,running), T>310 , T<334.

#show simpleEvent/3.