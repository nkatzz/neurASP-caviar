

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
orientation(fra2gt_3,150,298). coords(fra2gt_3,85,117,298).  visible(fra2gt_3,298). orientation(fra2gt_4,155,298). coords(fra2gt_4,108,121,298).  visible(fra2gt_4,298).
orientation(fra2gt_3,150,299). coords(fra2gt_3,83,117,299).  visible(fra2gt_3,299). orientation(fra2gt_4,155,299). coords(fra2gt_4,107,120,299).  visible(fra2gt_4,299).
orientation(fra2gt_3,150,300). coords(fra2gt_3,83,116,300).  visible(fra2gt_3,300). orientation(fra2gt_4,155,300). coords(fra2gt_4,101,120,300).  visible(fra2gt_4,300).
orientation(fra2gt_3,150,301). coords(fra2gt_3,84,116,301).  visible(fra2gt_3,301). orientation(fra2gt_4,155,301). coords(fra2gt_4,101,120,301).  visible(fra2gt_4,301).
orientation(fra2gt_3,150,302). coords(fra2gt_3,83,116,302).  visible(fra2gt_3,302). orientation(fra2gt_4,155,302). coords(fra2gt_4,100,120,302).  visible(fra2gt_4,302).
orientation(fra2gt_3,150,303). coords(fra2gt_3,82,116,303).  visible(fra2gt_3,303). orientation(fra2gt_4,155,303). coords(fra2gt_4,99,120,303).  visible(fra2gt_4,303).
orientation(fra2gt_3,150,304). coords(fra2gt_3,82,116,304).  visible(fra2gt_3,304). orientation(fra2gt_4,155,304). coords(fra2gt_4,100,123,304).  visible(fra2gt_4,304).
orientation(fra2gt_3,150,305). coords(fra2gt_3,84,112,305).  visible(fra2gt_3,305). orientation(fra2gt_4,155,305). coords(fra2gt_4,99,122,305).  visible(fra2gt_4,305).
orientation(fra2gt_3,150,306). coords(fra2gt_3,93,117,306).  visible(fra2gt_3,306). orientation(fra2gt_4,155,306). coords(fra2gt_4,99,123,306).  visible(fra2gt_4,306).
orientation(fra2gt_3,150,307). coords(fra2gt_3,93,117,307).  visible(fra2gt_3,307). orientation(fra2gt_4,155,307). coords(fra2gt_4,99,123,307).  visible(fra2gt_4,307).
orientation(fra2gt_3,150,308). coords(fra2gt_3,94,117,308).  visible(fra2gt_3,308). orientation(fra2gt_4,155,308). coords(fra2gt_4,98,122,308).  visible(fra2gt_4,308).
orientation(fra2gt_3,150,309). coords(fra2gt_3,95,117,309).  visible(fra2gt_3,309). orientation(fra2gt_4,155,309). coords(fra2gt_4,98,120,309).  visible(fra2gt_4,309).
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
orientation(fra2gt_3,150,323). coords(fra2gt_3,94,116,323).  visible(fra2gt_3,323). orientation(fra2gt_4,155,323). coords(fra2gt_4,101,121,323).  visible(fra2gt_4,323).
happensAt(disappear(fra2gt_3),322). happensAt(disappear(fra2gt_4),322).
1{simpleEvent(T,fra2gt_3,active); simpleEvent(T,fra2gt_3,inactive); simpleEvent(T,fra2gt_3,walking); simpleEvent(T,fra2gt_3,running)}1 :- T=298..322.
1{simpleEvent(T,fra2gt_4,active); simpleEvent(T,fra2gt_4,inactive); simpleEvent(T,fra2gt_4,walking); simpleEvent(T,fra2gt_4,running)}1 :- T=298..322.
:- holdsAt(moving(fra2gt_3,fra2gt_4),298). :- holdsAt(moving(fra2gt_4,fra2gt_3),298).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),299). :- not holdsAt(moving(fra2gt_4,fra2gt_3),299).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),300). :- not holdsAt(moving(fra2gt_4,fra2gt_3),300).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),301). :- not holdsAt(moving(fra2gt_4,fra2gt_3),301).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),302). :- not holdsAt(moving(fra2gt_4,fra2gt_3),302).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),303). :- not holdsAt(moving(fra2gt_4,fra2gt_3),303).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),304). :- not holdsAt(moving(fra2gt_4,fra2gt_3),304).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),305). :- not holdsAt(moving(fra2gt_4,fra2gt_3),305).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),306). :- not holdsAt(moving(fra2gt_4,fra2gt_3),306).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),307). :- not holdsAt(moving(fra2gt_4,fra2gt_3),307).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),308). :- not holdsAt(moving(fra2gt_4,fra2gt_3),308).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),309). :- not holdsAt(moving(fra2gt_4,fra2gt_3),309).
:- not holdsAt(moving(fra2gt_3,fra2gt_4),310). :- not holdsAt(moving(fra2gt_4,fra2gt_3),310).
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
:- holdsAt(moving(fra2gt_3,fra2gt_4),323). :- holdsAt(moving(fra2gt_4,fra2gt_3),323).
:- simpleEvent(T,P,inactive), T>298 , T<322.
:- simpleEvent(T,P,running), T>298 , T<322.

#show simpleEvent/3.