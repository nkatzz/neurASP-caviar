

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
orientation(wk1gt_4,157,291). coords(wk1gt_4,52,177,291).  visible(wk1gt_4,291). orientation(wk1gt_5,162,291). coords(wk1gt_5,50,165,291).  visible(wk1gt_5,291).
orientation(wk1gt_4,162,292). coords(wk1gt_4,52,177,292).  visible(wk1gt_4,292). orientation(wk1gt_5,162,292). coords(wk1gt_5,50,165,292).  visible(wk1gt_5,292).
orientation(wk1gt_4,161,293). coords(wk1gt_4,52,177,293).  visible(wk1gt_4,293). orientation(wk1gt_5,162,293). coords(wk1gt_5,51,165,293).  visible(wk1gt_5,293).
orientation(wk1gt_4,157,294). coords(wk1gt_4,52,177,294).  visible(wk1gt_4,294). orientation(wk1gt_5,162,294). coords(wk1gt_5,51,165,294).  visible(wk1gt_5,294).
orientation(wk1gt_4,158,295). coords(wk1gt_4,52,177,295).  visible(wk1gt_4,295). orientation(wk1gt_5,162,295). coords(wk1gt_5,50,164,295).  visible(wk1gt_5,295).
orientation(wk1gt_4,161,296). coords(wk1gt_4,52,176,296).  visible(wk1gt_4,296). orientation(wk1gt_5,162,296). coords(wk1gt_5,50,164,296).  visible(wk1gt_5,296).
orientation(wk1gt_4,160,297). coords(wk1gt_4,52,176,297).  visible(wk1gt_4,297). orientation(wk1gt_5,162,297). coords(wk1gt_5,50,164,297).  visible(wk1gt_5,297).
orientation(wk1gt_4,161,298). coords(wk1gt_4,52,175,298).  visible(wk1gt_4,298). orientation(wk1gt_5,162,298). coords(wk1gt_5,50,164,298).  visible(wk1gt_5,298).
orientation(wk1gt_4,167,299). coords(wk1gt_4,52,175,299).  visible(wk1gt_4,299). orientation(wk1gt_5,162,299). coords(wk1gt_5,50,164,299).  visible(wk1gt_5,299).
orientation(wk1gt_4,168,300). coords(wk1gt_4,52,174,300).  visible(wk1gt_4,300). orientation(wk1gt_5,162,300). coords(wk1gt_5,50,164,300).  visible(wk1gt_5,300).
orientation(wk1gt_4,173,301). coords(wk1gt_4,52,174,301).  visible(wk1gt_4,301). orientation(wk1gt_5,162,301). coords(wk1gt_5,50,164,301).  visible(wk1gt_5,301).
orientation(wk1gt_4,170,302). coords(wk1gt_4,52,173,302).  visible(wk1gt_4,302). orientation(wk1gt_5,162,302). coords(wk1gt_5,51,164,302).  visible(wk1gt_5,302).
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
orientation(wk1gt_4,159,316). coords(wk1gt_4,52,176,316).  visible(wk1gt_4,316). orientation(wk1gt_5,162,316). coords(wk1gt_5,50,163,316).  visible(wk1gt_5,316).
happensAt(disappear(wk1gt_4),315). happensAt(disappear(wk1gt_5),315).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=291..315.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=291..315.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),291). :- holdsAt(meeting(wk1gt_5,wk1gt_4),291).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),292). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),292).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),293). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),293).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),294). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),294).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),295). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),295).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),296). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),296).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),297). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),297).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),298). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),298).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),299). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),299).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),300). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),300).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),301). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),301).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),302). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),302).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),303). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),303).
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
:- holdsAt(meeting(wk1gt_4,wk1gt_5),316). :- holdsAt(meeting(wk1gt_5,wk1gt_4),316).
:- simpleEvent(T,P,walking), T>312 , T<312.
:- simpleEvent(T,P,running), T>312 , T<312.

#show simpleEvent/3.