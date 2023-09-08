

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
orientation(wk1gt_4,170,303). coords(wk1gt_4,52,173,303).  visible(wk1gt_4,303). orientation(wk1gt_5,162,303). coords(wk1gt_5,50,164,303).  visible(wk1gt_5,303).
orientation(wk1gt_4,170,304). coords(wk1gt_4,52,173,304).  visible(wk1gt_4,304). orientation(wk1gt_5,162,304). coords(wk1gt_5,50,163,304).  visible(wk1gt_5,304).
orientation(wk1gt_4,169,305). coords(wk1gt_4,52,173,305).  visible(wk1gt_4,305). orientation(wk1gt_5,162,305). coords(wk1gt_5,51,163,305).  visible(wk1gt_5,305).
orientation(wk1gt_4,168,306). coords(wk1gt_4,52,173,306).  visible(wk1gt_4,306). orientation(wk1gt_5,162,306). coords(wk1gt_5,51,163,306).  visible(wk1gt_5,306).
orientation(wk1gt_4,167,307). coords(wk1gt_4,52,173,307).  visible(wk1gt_4,307). orientation(wk1gt_5,162,307). coords(wk1gt_5,51,163,307).  visible(wk1gt_5,307).
orientation(wk1gt_4,167,308). coords(wk1gt_4,52,173,308).  visible(wk1gt_4,308). orientation(wk1gt_5,162,308). coords(wk1gt_5,50,163,308).  visible(wk1gt_5,308).
orientation(wk1gt_4,168,309). coords(wk1gt_4,52,174,309).  visible(wk1gt_4,309). orientation(wk1gt_5,162,309). coords(wk1gt_5,50,163,309).  visible(wk1gt_5,309).
orientation(wk1gt_4,164,310). coords(wk1gt_4,52,174,310).  visible(wk1gt_4,310). orientation(wk1gt_5,162,310). coords(wk1gt_5,50,163,310).  visible(wk1gt_5,310).
orientation(wk1gt_4,162,311). coords(wk1gt_4,52,175,311).  visible(wk1gt_4,311). orientation(wk1gt_5,162,311). coords(wk1gt_5,50,163,311).  visible(wk1gt_5,311).
orientation(wk1gt_4,161,312). coords(wk1gt_4,52,175,312).  visible(wk1gt_4,312). orientation(wk1gt_5,162,312). coords(wk1gt_5,50,163,312).  visible(wk1gt_5,312).
orientation(wk1gt_4,162,313). coords(wk1gt_4,52,175,313).  visible(wk1gt_4,313). orientation(wk1gt_5,162,313). coords(wk1gt_5,50,163,313).  visible(wk1gt_5,313).
orientation(wk1gt_4,161,314). coords(wk1gt_4,52,176,314).  visible(wk1gt_4,314). orientation(wk1gt_5,162,314). coords(wk1gt_5,51,163,314).  visible(wk1gt_5,314).
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
orientation(wk1gt_4,162,328). coords(wk1gt_4,52,176,328).  visible(wk1gt_4,328). orientation(wk1gt_5,162,328). coords(wk1gt_5,50,163,328).  visible(wk1gt_5,328).
happensAt(disappear(wk1gt_4),327). happensAt(disappear(wk1gt_5),327).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=303..327.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=303..327.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),303). :- holdsAt(meeting(wk1gt_5,wk1gt_4),303).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),304). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),304).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),305). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),305).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),306). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),306).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),307). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),307).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),308). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),308).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),309). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),309).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),310). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),310).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),311). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),311).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),312). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),312).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),313). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),313).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),314). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),314).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),315). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),315).
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
:- holdsAt(meeting(wk1gt_4,wk1gt_5),328). :- holdsAt(meeting(wk1gt_5,wk1gt_4),328).
:- simpleEvent(T,P,walking), T>315 , T<312.
:- simpleEvent(T,P,running), T>315 , T<312.

#show simpleEvent/3.