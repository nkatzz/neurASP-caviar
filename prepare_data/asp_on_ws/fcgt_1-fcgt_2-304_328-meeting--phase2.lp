

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

	person(fcgt_1;fcgt_2).
orientation(fcgt_1,153,304). coords(fcgt_1,98,127,304).  visible(fcgt_1,304). orientation(fcgt_2,150,304). coords(fcgt_2,92,130,304).  visible(fcgt_2,304).
orientation(fcgt_1,153,305). coords(fcgt_1,98,128,305).  visible(fcgt_1,305). orientation(fcgt_2,150,305). coords(fcgt_2,92,130,305).  visible(fcgt_2,305).
orientation(fcgt_1,153,306). coords(fcgt_1,97,129,306).  visible(fcgt_1,306). orientation(fcgt_2,150,306). coords(fcgt_2,92,129,306).  visible(fcgt_2,306).
orientation(fcgt_1,153,307). coords(fcgt_1,97,129,307).  visible(fcgt_1,307). orientation(fcgt_2,150,307). coords(fcgt_2,92,129,307).  visible(fcgt_2,307).
orientation(fcgt_1,153,308). coords(fcgt_1,98,129,308).  visible(fcgt_1,308). orientation(fcgt_2,150,308). coords(fcgt_2,92,129,308).  visible(fcgt_2,308).
orientation(fcgt_1,153,309). coords(fcgt_1,98,129,309).  visible(fcgt_1,309). orientation(fcgt_2,150,309). coords(fcgt_2,92,129,309).  visible(fcgt_2,309).
orientation(fcgt_1,153,310). coords(fcgt_1,98,129,310).  visible(fcgt_1,310). orientation(fcgt_2,150,310). coords(fcgt_2,93,129,310).  visible(fcgt_2,310).
orientation(fcgt_1,153,311). coords(fcgt_1,98,128,311).  visible(fcgt_1,311). orientation(fcgt_2,150,311). coords(fcgt_2,94,129,311).  visible(fcgt_2,311).
orientation(fcgt_1,153,312). coords(fcgt_1,98,128,312).  visible(fcgt_1,312). orientation(fcgt_2,150,312). coords(fcgt_2,93,129,312).  visible(fcgt_2,312).
orientation(fcgt_1,153,313). coords(fcgt_1,98,128,313).  visible(fcgt_1,313). orientation(fcgt_2,150,313). coords(fcgt_2,95,129,313).  visible(fcgt_2,313).
orientation(fcgt_1,153,314). coords(fcgt_1,98,130,314).  visible(fcgt_1,314). orientation(fcgt_2,150,314). coords(fcgt_2,95,129,314).  visible(fcgt_2,314).
orientation(fcgt_1,153,315). coords(fcgt_1,98,130,315).  visible(fcgt_1,315). orientation(fcgt_2,150,315). coords(fcgt_2,95,129,315).  visible(fcgt_2,315).
orientation(fcgt_1,153,316). coords(fcgt_1,98,131,316).  visible(fcgt_1,316). orientation(fcgt_2,150,316). coords(fcgt_2,95,129,316).  visible(fcgt_2,316).
orientation(fcgt_1,153,317). coords(fcgt_1,99,131,317).  visible(fcgt_1,317). orientation(fcgt_2,150,317). coords(fcgt_2,95,129,317).  visible(fcgt_2,317).
orientation(fcgt_1,153,318). coords(fcgt_1,100,130,318).  visible(fcgt_1,318). orientation(fcgt_2,150,318). coords(fcgt_2,95,129,318).  visible(fcgt_2,318).
orientation(fcgt_1,153,319). coords(fcgt_1,101,130,319).  visible(fcgt_1,319). orientation(fcgt_2,150,319). coords(fcgt_2,94,129,319).  visible(fcgt_2,319).
orientation(fcgt_1,153,320). coords(fcgt_1,102,130,320).  visible(fcgt_1,320). orientation(fcgt_2,150,320). coords(fcgt_2,94,129,320).  visible(fcgt_2,320).
orientation(fcgt_1,153,321). coords(fcgt_1,103,131,321).  visible(fcgt_1,321). orientation(fcgt_2,150,321). coords(fcgt_2,94,129,321).  visible(fcgt_2,321).
orientation(fcgt_1,153,322). coords(fcgt_1,104,133,322).  visible(fcgt_1,322). orientation(fcgt_2,150,322). coords(fcgt_2,94,129,322).  visible(fcgt_2,322).
orientation(fcgt_1,153,323). coords(fcgt_1,103,134,323).  visible(fcgt_1,323). orientation(fcgt_2,150,323). coords(fcgt_2,94,129,323).  visible(fcgt_2,323).
orientation(fcgt_1,153,324). coords(fcgt_1,103,135,324).  visible(fcgt_1,324). orientation(fcgt_2,150,324). coords(fcgt_2,94,129,324).  visible(fcgt_2,324).
orientation(fcgt_1,153,325). coords(fcgt_1,103,136,325).  visible(fcgt_1,325). orientation(fcgt_2,150,325). coords(fcgt_2,93,128,325).  visible(fcgt_2,325).
orientation(fcgt_1,155,326). coords(fcgt_1,103,136,326).  visible(fcgt_1,326). orientation(fcgt_2,146,326). coords(fcgt_2,93,128,326).  visible(fcgt_2,326).
orientation(fcgt_1,154,327). coords(fcgt_1,103,137,327).  visible(fcgt_1,327). orientation(fcgt_2,146,327). coords(fcgt_2,92,127,327).  visible(fcgt_2,327).
orientation(fcgt_1,156,328). coords(fcgt_1,103,137,328).  visible(fcgt_1,328). orientation(fcgt_2,146,328). coords(fcgt_2,91,127,328).  visible(fcgt_2,328).
orientation(fcgt_1,156,329). coords(fcgt_1,103,137,329).  visible(fcgt_1,329). orientation(fcgt_2,146,329). coords(fcgt_2,91,127,329).  visible(fcgt_2,329).
happensAt(disappear(fcgt_1),328). happensAt(disappear(fcgt_2),328).
1{simpleEvent(T,fcgt_1,active); simpleEvent(T,fcgt_1,inactive); simpleEvent(T,fcgt_1,walking); simpleEvent(T,fcgt_1,running)}1 :- T=304..328.
1{simpleEvent(T,fcgt_2,active); simpleEvent(T,fcgt_2,inactive); simpleEvent(T,fcgt_2,walking); simpleEvent(T,fcgt_2,running)}1 :- T=304..328.
:- holdsAt(meeting(fcgt_1,fcgt_2),304). :- holdsAt(meeting(fcgt_2,fcgt_1),304).
:- not holdsAt(meeting(fcgt_1,fcgt_2),305). :- not holdsAt(meeting(fcgt_2,fcgt_1),305).
:- not holdsAt(meeting(fcgt_1,fcgt_2),306). :- not holdsAt(meeting(fcgt_2,fcgt_1),306).
:- not holdsAt(meeting(fcgt_1,fcgt_2),307). :- not holdsAt(meeting(fcgt_2,fcgt_1),307).
:- not holdsAt(meeting(fcgt_1,fcgt_2),308). :- not holdsAt(meeting(fcgt_2,fcgt_1),308).
:- not holdsAt(meeting(fcgt_1,fcgt_2),309). :- not holdsAt(meeting(fcgt_2,fcgt_1),309).
:- not holdsAt(meeting(fcgt_1,fcgt_2),310). :- not holdsAt(meeting(fcgt_2,fcgt_1),310).
:- not holdsAt(meeting(fcgt_1,fcgt_2),311). :- not holdsAt(meeting(fcgt_2,fcgt_1),311).
:- not holdsAt(meeting(fcgt_1,fcgt_2),312). :- not holdsAt(meeting(fcgt_2,fcgt_1),312).
:- not holdsAt(meeting(fcgt_1,fcgt_2),313). :- not holdsAt(meeting(fcgt_2,fcgt_1),313).
:- not holdsAt(meeting(fcgt_1,fcgt_2),314). :- not holdsAt(meeting(fcgt_2,fcgt_1),314).
:- not holdsAt(meeting(fcgt_1,fcgt_2),315). :- not holdsAt(meeting(fcgt_2,fcgt_1),315).
:- not holdsAt(meeting(fcgt_1,fcgt_2),316). :- not holdsAt(meeting(fcgt_2,fcgt_1),316).
:- not holdsAt(meeting(fcgt_1,fcgt_2),317). :- not holdsAt(meeting(fcgt_2,fcgt_1),317).
:- not holdsAt(meeting(fcgt_1,fcgt_2),318). :- not holdsAt(meeting(fcgt_2,fcgt_1),318).
:- not holdsAt(meeting(fcgt_1,fcgt_2),319). :- not holdsAt(meeting(fcgt_2,fcgt_1),319).
:- not holdsAt(meeting(fcgt_1,fcgt_2),320). :- not holdsAt(meeting(fcgt_2,fcgt_1),320).
:- not holdsAt(meeting(fcgt_1,fcgt_2),321). :- not holdsAt(meeting(fcgt_2,fcgt_1),321).
:- not holdsAt(meeting(fcgt_1,fcgt_2),322). :- not holdsAt(meeting(fcgt_2,fcgt_1),322).
:- not holdsAt(meeting(fcgt_1,fcgt_2),323). :- not holdsAt(meeting(fcgt_2,fcgt_1),323).
:- not holdsAt(meeting(fcgt_1,fcgt_2),324). :- not holdsAt(meeting(fcgt_2,fcgt_1),324).
:- not holdsAt(meeting(fcgt_1,fcgt_2),325). :- not holdsAt(meeting(fcgt_2,fcgt_1),325).
:- not holdsAt(meeting(fcgt_1,fcgt_2),326). :- not holdsAt(meeting(fcgt_2,fcgt_1),326).
:- not holdsAt(meeting(fcgt_1,fcgt_2),327). :- not holdsAt(meeting(fcgt_2,fcgt_1),327).
:- not holdsAt(meeting(fcgt_1,fcgt_2),328). :- not holdsAt(meeting(fcgt_2,fcgt_1),328).
:- holdsAt(meeting(fcgt_1,fcgt_2),329). :- holdsAt(meeting(fcgt_2,fcgt_1),329).
:- simpleEvent(T,P,walking), T>304 , T<328.
:- simpleEvent(T,P,running), T>304 , T<328.

#show simpleEvent/3.