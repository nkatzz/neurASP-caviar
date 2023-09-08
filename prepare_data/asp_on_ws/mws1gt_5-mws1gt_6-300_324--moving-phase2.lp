

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

	person(mws1gt_5;mws1gt_6).
orientation(mws1gt_5,168,300). coords(mws1gt_5,36,178,300).  visible(mws1gt_5,300). orientation(mws1gt_6,146,300). coords(mws1gt_6,36,155,300).  visible(mws1gt_6,300).
orientation(mws1gt_5,168,301). coords(mws1gt_5,36,179,301).  visible(mws1gt_5,301). orientation(mws1gt_6,146,301). coords(mws1gt_6,36,155,301).  visible(mws1gt_6,301).
orientation(mws1gt_5,168,302). coords(mws1gt_5,36,179,302).  visible(mws1gt_5,302). orientation(mws1gt_6,146,302). coords(mws1gt_6,36,155,302).  visible(mws1gt_6,302).
orientation(mws1gt_5,168,303). coords(mws1gt_5,36,179,303).  visible(mws1gt_5,303). orientation(mws1gt_6,146,303). coords(mws1gt_6,36,156,303).  visible(mws1gt_6,303).
orientation(mws1gt_5,168,304). coords(mws1gt_5,36,179,304).  visible(mws1gt_5,304). orientation(mws1gt_6,146,304). coords(mws1gt_6,35,157,304).  visible(mws1gt_6,304).
orientation(mws1gt_5,168,305). coords(mws1gt_5,36,180,305).  visible(mws1gt_5,305). orientation(mws1gt_6,146,305). coords(mws1gt_6,35,159,305).  visible(mws1gt_6,305).
orientation(mws1gt_5,168,306). coords(mws1gt_5,36,181,306).  visible(mws1gt_5,306). orientation(mws1gt_6,146,306). coords(mws1gt_6,35,160,306).  visible(mws1gt_6,306).
orientation(mws1gt_5,168,307). coords(mws1gt_5,36,182,307).  visible(mws1gt_5,307). orientation(mws1gt_6,146,307). coords(mws1gt_6,35,162,307).  visible(mws1gt_6,307).
orientation(mws1gt_5,168,308). coords(mws1gt_5,35,183,308).  visible(mws1gt_5,308). orientation(mws1gt_6,146,308). coords(mws1gt_6,35,163,308).  visible(mws1gt_6,308).
orientation(mws1gt_5,168,309). coords(mws1gt_5,35,185,309).  visible(mws1gt_5,309). orientation(mws1gt_6,146,309). coords(mws1gt_6,35,164,309).  visible(mws1gt_6,309).
orientation(mws1gt_5,168,310). coords(mws1gt_5,35,186,310).  visible(mws1gt_5,310). orientation(mws1gt_6,146,310). coords(mws1gt_6,35,164,310).  visible(mws1gt_6,310).
orientation(mws1gt_5,168,311). coords(mws1gt_5,35,188,311).  visible(mws1gt_5,311). orientation(mws1gt_6,146,311). coords(mws1gt_6,35,165,311).  visible(mws1gt_6,311).
orientation(mws1gt_5,168,312). coords(mws1gt_5,35,189,312).  visible(mws1gt_5,312). orientation(mws1gt_6,146,312). coords(mws1gt_6,35,166,312).  visible(mws1gt_6,312).
orientation(mws1gt_5,168,313). coords(mws1gt_5,35,190,313).  visible(mws1gt_5,313). orientation(mws1gt_6,146,313). coords(mws1gt_6,35,166,313).  visible(mws1gt_6,313).
orientation(mws1gt_5,168,314). coords(mws1gt_5,35,190,314).  visible(mws1gt_5,314). orientation(mws1gt_6,146,314). coords(mws1gt_6,35,166,314).  visible(mws1gt_6,314).
orientation(mws1gt_5,168,315). coords(mws1gt_5,35,190,315).  visible(mws1gt_5,315). orientation(mws1gt_6,146,315). coords(mws1gt_6,35,167,315).  visible(mws1gt_6,315).
orientation(mws1gt_5,168,316). coords(mws1gt_5,35,191,316).  visible(mws1gt_5,316). orientation(mws1gt_6,146,316). coords(mws1gt_6,35,167,316).  visible(mws1gt_6,316).
orientation(mws1gt_5,168,317). coords(mws1gt_5,35,191,317).  visible(mws1gt_5,317). orientation(mws1gt_6,146,317). coords(mws1gt_6,35,167,317).  visible(mws1gt_6,317).
orientation(mws1gt_5,168,318). coords(mws1gt_5,35,193,318).  visible(mws1gt_5,318). orientation(mws1gt_6,146,318). coords(mws1gt_6,35,168,318).  visible(mws1gt_6,318).
orientation(mws1gt_5,168,319). coords(mws1gt_5,35,195,319).  visible(mws1gt_5,319). orientation(mws1gt_6,146,319). coords(mws1gt_6,35,169,319).  visible(mws1gt_6,319).
orientation(mws1gt_5,168,320). coords(mws1gt_5,35,197,320).  visible(mws1gt_5,320). orientation(mws1gt_6,146,320). coords(mws1gt_6,35,170,320).  visible(mws1gt_6,320).
orientation(mws1gt_5,168,321). coords(mws1gt_5,35,198,321).  visible(mws1gt_5,321). orientation(mws1gt_6,146,321). coords(mws1gt_6,34,172,321).  visible(mws1gt_6,321).
orientation(mws1gt_5,168,322). coords(mws1gt_5,35,200,322).  visible(mws1gt_5,322). orientation(mws1gt_6,146,322). coords(mws1gt_6,33,174,322).  visible(mws1gt_6,322).
orientation(mws1gt_5,168,323). coords(mws1gt_5,33,206,323).  visible(mws1gt_5,323). orientation(mws1gt_6,146,323). coords(mws1gt_6,33,176,323).  visible(mws1gt_6,323).
orientation(mws1gt_5,168,324). coords(mws1gt_5,33,206,324).  visible(mws1gt_5,324). orientation(mws1gt_6,146,324). coords(mws1gt_6,33,176,324).  visible(mws1gt_6,324).
orientation(mws1gt_5,168,325). coords(mws1gt_5,33,206,325).  visible(mws1gt_5,325). orientation(mws1gt_6,146,325). coords(mws1gt_6,33,176,325).  visible(mws1gt_6,325).
happensAt(disappear(mws1gt_5),324). happensAt(disappear(mws1gt_6),324).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=300..324.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=300..324.
:- holdsAt(moving(mws1gt_5,mws1gt_6),300). :- holdsAt(moving(mws1gt_6,mws1gt_5),300).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),301). :- not holdsAt(moving(mws1gt_6,mws1gt_5),301).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),302). :- not holdsAt(moving(mws1gt_6,mws1gt_5),302).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),303). :- not holdsAt(moving(mws1gt_6,mws1gt_5),303).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),304). :- not holdsAt(moving(mws1gt_6,mws1gt_5),304).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),305). :- not holdsAt(moving(mws1gt_6,mws1gt_5),305).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),306). :- not holdsAt(moving(mws1gt_6,mws1gt_5),306).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),307). :- not holdsAt(moving(mws1gt_6,mws1gt_5),307).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),308). :- not holdsAt(moving(mws1gt_6,mws1gt_5),308).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),309). :- not holdsAt(moving(mws1gt_6,mws1gt_5),309).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),310). :- not holdsAt(moving(mws1gt_6,mws1gt_5),310).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),311). :- not holdsAt(moving(mws1gt_6,mws1gt_5),311).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),312). :- not holdsAt(moving(mws1gt_6,mws1gt_5),312).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),313). :- not holdsAt(moving(mws1gt_6,mws1gt_5),313).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),314). :- not holdsAt(moving(mws1gt_6,mws1gt_5),314).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),315). :- not holdsAt(moving(mws1gt_6,mws1gt_5),315).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),316). :- not holdsAt(moving(mws1gt_6,mws1gt_5),316).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),317). :- not holdsAt(moving(mws1gt_6,mws1gt_5),317).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),318). :- not holdsAt(moving(mws1gt_6,mws1gt_5),318).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),319). :- not holdsAt(moving(mws1gt_6,mws1gt_5),319).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),320). :- not holdsAt(moving(mws1gt_6,mws1gt_5),320).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),321). :- not holdsAt(moving(mws1gt_6,mws1gt_5),321).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),322). :- not holdsAt(moving(mws1gt_6,mws1gt_5),322).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),323). :- not holdsAt(moving(mws1gt_6,mws1gt_5),323).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),324). :- not holdsAt(moving(mws1gt_6,mws1gt_5),324).
:- holdsAt(moving(mws1gt_5,mws1gt_6),325). :- holdsAt(moving(mws1gt_6,mws1gt_5),325).
:- simpleEvent(T,P,inactive), T>300 , T<324.
:- simpleEvent(T,P,running), T>300 , T<324.

#show simpleEvent/3.