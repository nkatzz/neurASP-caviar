

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
orientation(fcgt_1,153,292). coords(fcgt_1,107,128,292).  visible(fcgt_1,292). orientation(fcgt_2,150,292). coords(fcgt_2,85,133,292).  visible(fcgt_2,292).
orientation(fcgt_1,153,293). coords(fcgt_1,107,126,293).  visible(fcgt_1,293). orientation(fcgt_2,150,293). coords(fcgt_2,85,133,293).  visible(fcgt_2,293).
orientation(fcgt_1,153,294). coords(fcgt_1,104,126,294).  visible(fcgt_1,294). orientation(fcgt_2,150,294). coords(fcgt_2,86,132,294).  visible(fcgt_2,294).
orientation(fcgt_1,153,295). coords(fcgt_1,103,127,295).  visible(fcgt_1,295). orientation(fcgt_2,150,295). coords(fcgt_2,86,132,295).  visible(fcgt_2,295).
orientation(fcgt_1,153,296). coords(fcgt_1,101,125,296).  visible(fcgt_1,296). orientation(fcgt_2,150,296). coords(fcgt_2,86,131,296).  visible(fcgt_2,296).
orientation(fcgt_1,153,297). coords(fcgt_1,101,124,297).  visible(fcgt_1,297). orientation(fcgt_2,150,297). coords(fcgt_2,86,131,297).  visible(fcgt_2,297).
orientation(fcgt_1,153,298). coords(fcgt_1,101,123,298).  visible(fcgt_1,298). orientation(fcgt_2,150,298). coords(fcgt_2,87,131,298).  visible(fcgt_2,298).
orientation(fcgt_1,153,299). coords(fcgt_1,100,123,299).  visible(fcgt_1,299). orientation(fcgt_2,150,299). coords(fcgt_2,88,131,299).  visible(fcgt_2,299).
orientation(fcgt_1,153,300). coords(fcgt_1,100,123,300).  visible(fcgt_1,300). orientation(fcgt_2,150,300). coords(fcgt_2,90,130,300).  visible(fcgt_2,300).
orientation(fcgt_1,153,301). coords(fcgt_1,99,123,301).  visible(fcgt_1,301). orientation(fcgt_2,150,301). coords(fcgt_2,91,130,301).  visible(fcgt_2,301).
orientation(fcgt_1,153,302). coords(fcgt_1,99,124,302).  visible(fcgt_1,302). orientation(fcgt_2,150,302). coords(fcgt_2,91,130,302).  visible(fcgt_2,302).
orientation(fcgt_1,153,303). coords(fcgt_1,98,125,303).  visible(fcgt_1,303). orientation(fcgt_2,150,303). coords(fcgt_2,92,130,303).  visible(fcgt_2,303).
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
orientation(fcgt_1,153,317). coords(fcgt_1,98,131,317).  visible(fcgt_1,317). orientation(fcgt_2,150,317). coords(fcgt_2,95,129,317).  visible(fcgt_2,317).
happensAt(disappear(fcgt_1),316). happensAt(disappear(fcgt_2),316).
1{simpleEvent(T,fcgt_1,active); simpleEvent(T,fcgt_1,inactive); simpleEvent(T,fcgt_1,walking); simpleEvent(T,fcgt_1,running)}1 :- T=292..316.
1{simpleEvent(T,fcgt_2,active); simpleEvent(T,fcgt_2,inactive); simpleEvent(T,fcgt_2,walking); simpleEvent(T,fcgt_2,running)}1 :- T=292..316.
:- holdsAt(moving(fcgt_1,fcgt_2),292). :- holdsAt(moving(fcgt_2,fcgt_1),292).
:- not holdsAt(moving(fcgt_1,fcgt_2),293). :- not holdsAt(moving(fcgt_2,fcgt_1),293).
:- not holdsAt(moving(fcgt_1,fcgt_2),294). :- not holdsAt(moving(fcgt_2,fcgt_1),294).
:- not holdsAt(moving(fcgt_1,fcgt_2),295). :- not holdsAt(moving(fcgt_2,fcgt_1),295).
:- not holdsAt(moving(fcgt_1,fcgt_2),296). :- not holdsAt(moving(fcgt_2,fcgt_1),296).
:- not holdsAt(moving(fcgt_1,fcgt_2),297). :- not holdsAt(moving(fcgt_2,fcgt_1),297).
:- holdsAt(moving(fcgt_1,fcgt_2),298). :- holdsAt(moving(fcgt_2,fcgt_1),298).
:- holdsAt(meeting(fcgt_1,fcgt_2),297). :- holdsAt(meeting(fcgt_2,fcgt_1),297).
:- not holdsAt(meeting(fcgt_1,fcgt_2),299). :- not holdsAt(meeting(fcgt_2,fcgt_1),299).
:- not holdsAt(meeting(fcgt_1,fcgt_2),300). :- not holdsAt(meeting(fcgt_2,fcgt_1),300).
:- not holdsAt(meeting(fcgt_1,fcgt_2),301). :- not holdsAt(meeting(fcgt_2,fcgt_1),301).
:- not holdsAt(meeting(fcgt_1,fcgt_2),302). :- not holdsAt(meeting(fcgt_2,fcgt_1),302).
:- not holdsAt(meeting(fcgt_1,fcgt_2),303). :- not holdsAt(meeting(fcgt_2,fcgt_1),303).
:- not holdsAt(meeting(fcgt_1,fcgt_2),304). :- not holdsAt(meeting(fcgt_2,fcgt_1),304).
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
:- holdsAt(meeting(fcgt_1,fcgt_2),317). :- holdsAt(meeting(fcgt_2,fcgt_1),317).
:- simpleEvent(T,P,walking), T>297 , T<316.
:- simpleEvent(T,P,running), T>297 , T<316.
:- simpleEvent(T,P,inactive), T>292 , T<297.
:- simpleEvent(T,P,running), T>292 , T<297.

#show simpleEvent/3.