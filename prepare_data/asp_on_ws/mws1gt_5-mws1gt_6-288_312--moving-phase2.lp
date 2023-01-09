

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
orientation(mws1gt_5,161,288). coords(mws1gt_5,37,166,288).  visible(mws1gt_5,288). orientation(mws1gt_6,146,288). coords(mws1gt_6,38,146,288).  visible(mws1gt_6,288).
orientation(mws1gt_5,161,289). coords(mws1gt_5,37,167,289).  visible(mws1gt_5,289). orientation(mws1gt_6,146,289). coords(mws1gt_6,37,146,289).  visible(mws1gt_6,289).
orientation(mws1gt_5,161,290). coords(mws1gt_5,37,168,290).  visible(mws1gt_5,290). orientation(mws1gt_6,146,290). coords(mws1gt_6,37,147,290).  visible(mws1gt_6,290).
orientation(mws1gt_5,168,291). coords(mws1gt_5,37,169,291).  visible(mws1gt_5,291). orientation(mws1gt_6,146,291). coords(mws1gt_6,37,148,291).  visible(mws1gt_6,291).
orientation(mws1gt_5,168,292). coords(mws1gt_5,37,170,292).  visible(mws1gt_5,292). orientation(mws1gt_6,146,292). coords(mws1gt_6,36,150,292).  visible(mws1gt_6,292).
orientation(mws1gt_5,168,293). coords(mws1gt_5,37,171,293).  visible(mws1gt_5,293). orientation(mws1gt_6,146,293). coords(mws1gt_6,36,151,293).  visible(mws1gt_6,293).
orientation(mws1gt_5,168,294). coords(mws1gt_5,37,172,294).  visible(mws1gt_5,294). orientation(mws1gt_6,146,294). coords(mws1gt_6,36,152,294).  visible(mws1gt_6,294).
orientation(mws1gt_5,168,295). coords(mws1gt_5,37,174,295).  visible(mws1gt_5,295). orientation(mws1gt_6,146,295). coords(mws1gt_6,36,152,295).  visible(mws1gt_6,295).
orientation(mws1gt_5,168,296). coords(mws1gt_5,36,175,296).  visible(mws1gt_5,296). orientation(mws1gt_6,146,296). coords(mws1gt_6,36,153,296).  visible(mws1gt_6,296).
orientation(mws1gt_5,168,297). coords(mws1gt_5,36,176,297).  visible(mws1gt_5,297). orientation(mws1gt_6,146,297). coords(mws1gt_6,36,153,297).  visible(mws1gt_6,297).
orientation(mws1gt_5,168,298). coords(mws1gt_5,36,177,298).  visible(mws1gt_5,298). orientation(mws1gt_6,146,298). coords(mws1gt_6,36,154,298).  visible(mws1gt_6,298).
orientation(mws1gt_5,168,299). coords(mws1gt_5,36,178,299).  visible(mws1gt_5,299). orientation(mws1gt_6,146,299). coords(mws1gt_6,36,155,299).  visible(mws1gt_6,299).
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
orientation(mws1gt_5,168,313). coords(mws1gt_5,35,189,313).  visible(mws1gt_5,313). orientation(mws1gt_6,146,313). coords(mws1gt_6,35,166,313).  visible(mws1gt_6,313).
happensAt(disappear(mws1gt_5),312). happensAt(disappear(mws1gt_6),312).
1{simpleEvent(T,mws1gt_5,active); simpleEvent(T,mws1gt_5,inactive); simpleEvent(T,mws1gt_5,walking); simpleEvent(T,mws1gt_5,running)}1 :- T=288..312.
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=288..312.
:- holdsAt(moving(mws1gt_5,mws1gt_6),288). :- holdsAt(moving(mws1gt_6,mws1gt_5),288).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),289). :- not holdsAt(moving(mws1gt_6,mws1gt_5),289).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),290). :- not holdsAt(moving(mws1gt_6,mws1gt_5),290).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),291). :- not holdsAt(moving(mws1gt_6,mws1gt_5),291).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),292). :- not holdsAt(moving(mws1gt_6,mws1gt_5),292).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),293). :- not holdsAt(moving(mws1gt_6,mws1gt_5),293).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),294). :- not holdsAt(moving(mws1gt_6,mws1gt_5),294).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),295). :- not holdsAt(moving(mws1gt_6,mws1gt_5),295).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),296). :- not holdsAt(moving(mws1gt_6,mws1gt_5),296).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),297). :- not holdsAt(moving(mws1gt_6,mws1gt_5),297).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),298). :- not holdsAt(moving(mws1gt_6,mws1gt_5),298).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),299). :- not holdsAt(moving(mws1gt_6,mws1gt_5),299).
:- not holdsAt(moving(mws1gt_5,mws1gt_6),300). :- not holdsAt(moving(mws1gt_6,mws1gt_5),300).
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
:- holdsAt(moving(mws1gt_5,mws1gt_6),313). :- holdsAt(moving(mws1gt_6,mws1gt_5),313).
:- simpleEvent(T,P,inactive), T>288 , T<312.
:- simpleEvent(T,P,running), T>288 , T<312.

#show simpleEvent/3.