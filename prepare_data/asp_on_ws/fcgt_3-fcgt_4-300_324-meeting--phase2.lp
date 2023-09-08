

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

	person(fcgt_3;fcgt_4).
orientation(fcgt_3,164,300). coords(fcgt_3,98,187,300).  visible(fcgt_3,300). orientation(fcgt_4,166,300). coords(fcgt_4,82,206,300).  visible(fcgt_4,300).
orientation(fcgt_3,164,301). coords(fcgt_3,98,187,301).  visible(fcgt_3,301). orientation(fcgt_4,166,301). coords(fcgt_4,82,206,301).  visible(fcgt_4,301).
orientation(fcgt_3,164,302). coords(fcgt_3,98,187,302).  visible(fcgt_3,302). orientation(fcgt_4,166,302). coords(fcgt_4,82,206,302).  visible(fcgt_4,302).
orientation(fcgt_3,164,303). coords(fcgt_3,98,187,303).  visible(fcgt_3,303). orientation(fcgt_4,166,303). coords(fcgt_4,82,206,303).  visible(fcgt_4,303).
orientation(fcgt_3,164,304). coords(fcgt_3,98,187,304).  visible(fcgt_3,304). orientation(fcgt_4,166,304). coords(fcgt_4,82,206,304).  visible(fcgt_4,304).
orientation(fcgt_3,164,305). coords(fcgt_3,98,187,305).  visible(fcgt_3,305). orientation(fcgt_4,166,305). coords(fcgt_4,82,206,305).  visible(fcgt_4,305).
orientation(fcgt_3,164,306). coords(fcgt_3,98,187,306).  visible(fcgt_3,306). orientation(fcgt_4,166,306). coords(fcgt_4,82,206,306).  visible(fcgt_4,306).
orientation(fcgt_3,164,307). coords(fcgt_3,98,187,307).  visible(fcgt_3,307). orientation(fcgt_4,166,307). coords(fcgt_4,82,206,307).  visible(fcgt_4,307).
orientation(fcgt_3,164,308). coords(fcgt_3,98,187,308).  visible(fcgt_3,308). orientation(fcgt_4,166,308). coords(fcgt_4,82,206,308).  visible(fcgt_4,308).
orientation(fcgt_3,164,309). coords(fcgt_3,98,187,309).  visible(fcgt_3,309). orientation(fcgt_4,166,309). coords(fcgt_4,82,206,309).  visible(fcgt_4,309).
orientation(fcgt_3,164,310). coords(fcgt_3,98,187,310).  visible(fcgt_3,310). orientation(fcgt_4,166,310). coords(fcgt_4,82,206,310).  visible(fcgt_4,310).
orientation(fcgt_3,164,311). coords(fcgt_3,98,187,311).  visible(fcgt_3,311). orientation(fcgt_4,166,311). coords(fcgt_4,82,206,311).  visible(fcgt_4,311).
orientation(fcgt_3,164,312). coords(fcgt_3,98,188,312).  visible(fcgt_3,312). orientation(fcgt_4,166,312). coords(fcgt_4,82,206,312).  visible(fcgt_4,312).
orientation(fcgt_3,164,313). coords(fcgt_3,98,188,313).  visible(fcgt_3,313). orientation(fcgt_4,166,313). coords(fcgt_4,82,206,313).  visible(fcgt_4,313).
orientation(fcgt_3,164,314). coords(fcgt_3,98,188,314).  visible(fcgt_3,314). orientation(fcgt_4,166,314). coords(fcgt_4,82,206,314).  visible(fcgt_4,314).
orientation(fcgt_3,164,315). coords(fcgt_3,98,188,315).  visible(fcgt_3,315). orientation(fcgt_4,166,315). coords(fcgt_4,82,206,315).  visible(fcgt_4,315).
orientation(fcgt_3,164,316). coords(fcgt_3,98,188,316).  visible(fcgt_3,316). orientation(fcgt_4,166,316). coords(fcgt_4,82,206,316).  visible(fcgt_4,316).
orientation(fcgt_3,164,317). coords(fcgt_3,98,188,317).  visible(fcgt_3,317). orientation(fcgt_4,166,317). coords(fcgt_4,82,206,317).  visible(fcgt_4,317).
orientation(fcgt_3,164,318). coords(fcgt_3,99,188,318).  visible(fcgt_3,318). orientation(fcgt_4,166,318). coords(fcgt_4,82,206,318).  visible(fcgt_4,318).
orientation(fcgt_3,164,319). coords(fcgt_3,98,187,319).  visible(fcgt_3,319). orientation(fcgt_4,166,319). coords(fcgt_4,82,206,319).  visible(fcgt_4,319).
orientation(fcgt_3,164,320). coords(fcgt_3,99,187,320).  visible(fcgt_3,320). orientation(fcgt_4,166,320). coords(fcgt_4,82,206,320).  visible(fcgt_4,320).
orientation(fcgt_3,164,321). coords(fcgt_3,98,187,321).  visible(fcgt_3,321). orientation(fcgt_4,166,321). coords(fcgt_4,82,206,321).  visible(fcgt_4,321).
orientation(fcgt_3,164,322). coords(fcgt_3,98,187,322).  visible(fcgt_3,322). orientation(fcgt_4,166,322). coords(fcgt_4,82,206,322).  visible(fcgt_4,322).
orientation(fcgt_3,164,323). coords(fcgt_3,98,187,323).  visible(fcgt_3,323). orientation(fcgt_4,166,323). coords(fcgt_4,82,206,323).  visible(fcgt_4,323).
orientation(fcgt_3,164,324). coords(fcgt_3,98,187,324).  visible(fcgt_3,324). orientation(fcgt_4,166,324). coords(fcgt_4,82,206,324).  visible(fcgt_4,324).
orientation(fcgt_3,164,325). coords(fcgt_3,98,187,325).  visible(fcgt_3,325). orientation(fcgt_4,166,325). coords(fcgt_4,82,206,325).  visible(fcgt_4,325).
happensAt(disappear(fcgt_3),324). happensAt(disappear(fcgt_4),324).
1{simpleEvent(T,fcgt_3,active); simpleEvent(T,fcgt_3,inactive); simpleEvent(T,fcgt_3,walking); simpleEvent(T,fcgt_3,running)}1 :- T=300..324.
1{simpleEvent(T,fcgt_4,active); simpleEvent(T,fcgt_4,inactive); simpleEvent(T,fcgt_4,walking); simpleEvent(T,fcgt_4,running)}1 :- T=300..324.
:- holdsAt(meeting(fcgt_3,fcgt_4),300). :- holdsAt(meeting(fcgt_4,fcgt_3),300).
:- not holdsAt(meeting(fcgt_3,fcgt_4),301). :- not holdsAt(meeting(fcgt_4,fcgt_3),301).
:- not holdsAt(meeting(fcgt_3,fcgt_4),302). :- not holdsAt(meeting(fcgt_4,fcgt_3),302).
:- not holdsAt(meeting(fcgt_3,fcgt_4),303). :- not holdsAt(meeting(fcgt_4,fcgt_3),303).
:- not holdsAt(meeting(fcgt_3,fcgt_4),304). :- not holdsAt(meeting(fcgt_4,fcgt_3),304).
:- not holdsAt(meeting(fcgt_3,fcgt_4),305). :- not holdsAt(meeting(fcgt_4,fcgt_3),305).
:- not holdsAt(meeting(fcgt_3,fcgt_4),306). :- not holdsAt(meeting(fcgt_4,fcgt_3),306).
:- not holdsAt(meeting(fcgt_3,fcgt_4),307). :- not holdsAt(meeting(fcgt_4,fcgt_3),307).
:- not holdsAt(meeting(fcgt_3,fcgt_4),308). :- not holdsAt(meeting(fcgt_4,fcgt_3),308).
:- not holdsAt(meeting(fcgt_3,fcgt_4),309). :- not holdsAt(meeting(fcgt_4,fcgt_3),309).
:- not holdsAt(meeting(fcgt_3,fcgt_4),310). :- not holdsAt(meeting(fcgt_4,fcgt_3),310).
:- not holdsAt(meeting(fcgt_3,fcgt_4),311). :- not holdsAt(meeting(fcgt_4,fcgt_3),311).
:- not holdsAt(meeting(fcgt_3,fcgt_4),312). :- not holdsAt(meeting(fcgt_4,fcgt_3),312).
:- not holdsAt(meeting(fcgt_3,fcgt_4),313). :- not holdsAt(meeting(fcgt_4,fcgt_3),313).
:- not holdsAt(meeting(fcgt_3,fcgt_4),314). :- not holdsAt(meeting(fcgt_4,fcgt_3),314).
:- not holdsAt(meeting(fcgt_3,fcgt_4),315). :- not holdsAt(meeting(fcgt_4,fcgt_3),315).
:- not holdsAt(meeting(fcgt_3,fcgt_4),316). :- not holdsAt(meeting(fcgt_4,fcgt_3),316).
:- not holdsAt(meeting(fcgt_3,fcgt_4),317). :- not holdsAt(meeting(fcgt_4,fcgt_3),317).
:- not holdsAt(meeting(fcgt_3,fcgt_4),318). :- not holdsAt(meeting(fcgt_4,fcgt_3),318).
:- not holdsAt(meeting(fcgt_3,fcgt_4),319). :- not holdsAt(meeting(fcgt_4,fcgt_3),319).
:- not holdsAt(meeting(fcgt_3,fcgt_4),320). :- not holdsAt(meeting(fcgt_4,fcgt_3),320).
:- not holdsAt(meeting(fcgt_3,fcgt_4),321). :- not holdsAt(meeting(fcgt_4,fcgt_3),321).
:- not holdsAt(meeting(fcgt_3,fcgt_4),322). :- not holdsAt(meeting(fcgt_4,fcgt_3),322).
:- not holdsAt(meeting(fcgt_3,fcgt_4),323). :- not holdsAt(meeting(fcgt_4,fcgt_3),323).
:- not holdsAt(meeting(fcgt_3,fcgt_4),324). :- not holdsAt(meeting(fcgt_4,fcgt_3),324).
:- holdsAt(meeting(fcgt_3,fcgt_4),325). :- holdsAt(meeting(fcgt_4,fcgt_3),325).
:- simpleEvent(T,P,walking), T>300 , T<324.
:- simpleEvent(T,P,running), T>300 , T<324.

#show simpleEvent/3.