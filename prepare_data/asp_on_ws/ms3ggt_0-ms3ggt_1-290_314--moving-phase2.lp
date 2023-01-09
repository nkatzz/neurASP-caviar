

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

	person(ms3ggt_0;ms3ggt_1).
orientation(ms3ggt_0,159,290). coords(ms3ggt_0,76,146,290).  visible(ms3ggt_0,290). orientation(ms3ggt_1,158,290). coords(ms3ggt_1,72,164,290).  visible(ms3ggt_1,290).
orientation(ms3ggt_0,159,291). coords(ms3ggt_0,76,146,291).  visible(ms3ggt_0,291). orientation(ms3ggt_1,158,291). coords(ms3ggt_1,72,164,291).  visible(ms3ggt_1,291).
orientation(ms3ggt_0,159,292). coords(ms3ggt_0,76,146,292).  visible(ms3ggt_0,292). orientation(ms3ggt_1,158,292). coords(ms3ggt_1,73,164,292).  visible(ms3ggt_1,292).
orientation(ms3ggt_0,159,293). coords(ms3ggt_0,77,146,293).  visible(ms3ggt_0,293). orientation(ms3ggt_1,158,293). coords(ms3ggt_1,75,164,293).  visible(ms3ggt_1,293).
orientation(ms3ggt_0,159,294). coords(ms3ggt_0,77,146,294).  visible(ms3ggt_0,294). orientation(ms3ggt_1,158,294). coords(ms3ggt_1,76,164,294).  visible(ms3ggt_1,294).
orientation(ms3ggt_0,159,295). coords(ms3ggt_0,77,146,295).  visible(ms3ggt_0,295). orientation(ms3ggt_1,158,295). coords(ms3ggt_1,76,164,295).  visible(ms3ggt_1,295).
orientation(ms3ggt_0,159,296). coords(ms3ggt_0,77,145,296).  visible(ms3ggt_0,296). orientation(ms3ggt_1,158,296). coords(ms3ggt_1,76,164,296).  visible(ms3ggt_1,296).
orientation(ms3ggt_0,159,297). coords(ms3ggt_0,77,145,297).  visible(ms3ggt_0,297). orientation(ms3ggt_1,158,297). coords(ms3ggt_1,76,164,297).  visible(ms3ggt_1,297).
orientation(ms3ggt_0,159,298). coords(ms3ggt_0,77,145,298).  visible(ms3ggt_0,298). orientation(ms3ggt_1,158,298). coords(ms3ggt_1,76,164,298).  visible(ms3ggt_1,298).
orientation(ms3ggt_0,159,299). coords(ms3ggt_0,77,145,299).  visible(ms3ggt_0,299). orientation(ms3ggt_1,158,299). coords(ms3ggt_1,77,164,299).  visible(ms3ggt_1,299).
orientation(ms3ggt_0,159,300). coords(ms3ggt_0,78,145,300).  visible(ms3ggt_0,300). orientation(ms3ggt_1,158,300). coords(ms3ggt_1,77,164,300).  visible(ms3ggt_1,300).
orientation(ms3ggt_0,159,301). coords(ms3ggt_0,80,144,301).  visible(ms3ggt_0,301). orientation(ms3ggt_1,158,301). coords(ms3ggt_1,77,164,301).  visible(ms3ggt_1,301).
orientation(ms3ggt_0,159,302). coords(ms3ggt_0,81,144,302).  visible(ms3ggt_0,302). orientation(ms3ggt_1,158,302). coords(ms3ggt_1,78,164,302).  visible(ms3ggt_1,302).
orientation(ms3ggt_0,159,303). coords(ms3ggt_0,82,144,303).  visible(ms3ggt_0,303). orientation(ms3ggt_1,158,303). coords(ms3ggt_1,78,164,303).  visible(ms3ggt_1,303).
orientation(ms3ggt_0,159,304). coords(ms3ggt_0,83,144,304).  visible(ms3ggt_0,304). orientation(ms3ggt_1,158,304). coords(ms3ggt_1,78,164,304).  visible(ms3ggt_1,304).
orientation(ms3ggt_0,159,305). coords(ms3ggt_0,84,144,305).  visible(ms3ggt_0,305). orientation(ms3ggt_1,158,305). coords(ms3ggt_1,79,164,305).  visible(ms3ggt_1,305).
orientation(ms3ggt_0,161,306). coords(ms3ggt_0,84,144,306).  visible(ms3ggt_0,306). orientation(ms3ggt_1,158,306). coords(ms3ggt_1,79,164,306).  visible(ms3ggt_1,306).
orientation(ms3ggt_0,161,307). coords(ms3ggt_0,85,144,307).  visible(ms3ggt_0,307). orientation(ms3ggt_1,160,307). coords(ms3ggt_1,81,164,307).  visible(ms3ggt_1,307).
orientation(ms3ggt_0,161,308). coords(ms3ggt_0,85,144,308).  visible(ms3ggt_0,308). orientation(ms3ggt_1,160,308). coords(ms3ggt_1,82,164,308).  visible(ms3ggt_1,308).
orientation(ms3ggt_0,161,309). coords(ms3ggt_0,86,144,309).  visible(ms3ggt_0,309). orientation(ms3ggt_1,157,309). coords(ms3ggt_1,84,164,309).  visible(ms3ggt_1,309).
orientation(ms3ggt_0,161,310). coords(ms3ggt_0,86,144,310).  visible(ms3ggt_0,310). orientation(ms3ggt_1,157,310). coords(ms3ggt_1,84,164,310).  visible(ms3ggt_1,310).
orientation(ms3ggt_0,158,311). coords(ms3ggt_0,86,144,311).  visible(ms3ggt_0,311). orientation(ms3ggt_1,160,311). coords(ms3ggt_1,84,165,311).  visible(ms3ggt_1,311).
orientation(ms3ggt_0,158,312). coords(ms3ggt_0,86,143,312).  visible(ms3ggt_0,312). orientation(ms3ggt_1,160,312). coords(ms3ggt_1,84,165,312).  visible(ms3ggt_1,312).
orientation(ms3ggt_0,158,313). coords(ms3ggt_0,86,143,313).  visible(ms3ggt_0,313). orientation(ms3ggt_1,160,313). coords(ms3ggt_1,84,165,313).  visible(ms3ggt_1,313).
orientation(ms3ggt_0,158,314). coords(ms3ggt_0,86,144,314).  visible(ms3ggt_0,314). orientation(ms3ggt_1,160,314). coords(ms3ggt_1,84,166,314).  visible(ms3ggt_1,314).
orientation(ms3ggt_0,158,315). coords(ms3ggt_0,86,144,315).  visible(ms3ggt_0,315). orientation(ms3ggt_1,160,315). coords(ms3ggt_1,84,166,315).  visible(ms3ggt_1,315).
happensAt(disappear(ms3ggt_0),314). happensAt(disappear(ms3ggt_1),314).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=290..314.
1{simpleEvent(T,ms3ggt_1,active); simpleEvent(T,ms3ggt_1,inactive); simpleEvent(T,ms3ggt_1,walking); simpleEvent(T,ms3ggt_1,running)}1 :- T=290..314.
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),290). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),290).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),291). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),291).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),292). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),292).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),293). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),293).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),294). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),294).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),295). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),295).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),296). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),296).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),297). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),297).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),298). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),298).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),299). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),299).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),300). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),300).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),301). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),301).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),302). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),302).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),303). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),303).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),304). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),304).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),305). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),305).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),306). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),306).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),307). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),307).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),308). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),308).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),309). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),309).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),310). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),310).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),311). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),311).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),312). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),312).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),313). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),313).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),314). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),314).
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),315). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),315).
:- simpleEvent(T,P,inactive), T>290 , T<314.
:- simpleEvent(T,P,running), T>290 , T<314.

#show simpleEvent/3.