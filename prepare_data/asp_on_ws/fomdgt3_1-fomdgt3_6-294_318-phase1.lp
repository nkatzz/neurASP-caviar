
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fomdgt3_1,150,294). coords(fomdgt3_1,46,96,294).  visible(fomdgt3_1,294). happensAt(active(fomdgt3_1),294). orientation(fomdgt3_6,130,294). coords(fomdgt3_6,217,133,294).  visible(fomdgt3_6,294). happensAt(walking(fomdgt3_6),294).
orientation(fomdgt3_1,150,295). coords(fomdgt3_1,46,96,295).  visible(fomdgt3_1,295). happensAt(active(fomdgt3_1),295). orientation(fomdgt3_6,130,295). coords(fomdgt3_6,217,130,295).  visible(fomdgt3_6,295). happensAt(walking(fomdgt3_6),295).
orientation(fomdgt3_1,150,296). coords(fomdgt3_1,46,96,296).  visible(fomdgt3_1,296). happensAt(inactive(fomdgt3_1),296). orientation(fomdgt3_6,130,296). coords(fomdgt3_6,217,127,296).  visible(fomdgt3_6,296). happensAt(walking(fomdgt3_6),296).
orientation(fomdgt3_1,150,297). coords(fomdgt3_1,46,96,297).  visible(fomdgt3_1,297). happensAt(inactive(fomdgt3_1),297). orientation(fomdgt3_6,130,297). coords(fomdgt3_6,217,125,297).  visible(fomdgt3_6,297). happensAt(walking(fomdgt3_6),297).
orientation(fomdgt3_1,150,298). coords(fomdgt3_1,46,96,298).  visible(fomdgt3_1,298). happensAt(inactive(fomdgt3_1),298). orientation(fomdgt3_6,130,298). coords(fomdgt3_6,217,124,298).  visible(fomdgt3_6,298). happensAt(walking(fomdgt3_6),298).
orientation(fomdgt3_1,150,299). coords(fomdgt3_1,46,96,299).  visible(fomdgt3_1,299). happensAt(inactive(fomdgt3_1),299). orientation(fomdgt3_6,130,299). coords(fomdgt3_6,217,123,299).  visible(fomdgt3_6,299). happensAt(walking(fomdgt3_6),299).
orientation(fomdgt3_1,150,300). coords(fomdgt3_1,46,96,300).  visible(fomdgt3_1,300). happensAt(inactive(fomdgt3_1),300). orientation(fomdgt3_6,130,300). coords(fomdgt3_6,217,123,300).  visible(fomdgt3_6,300). happensAt(walking(fomdgt3_6),300).
orientation(fomdgt3_1,150,301). coords(fomdgt3_1,46,96,301).  visible(fomdgt3_1,301). happensAt(inactive(fomdgt3_1),301). orientation(fomdgt3_6,130,301). coords(fomdgt3_6,216,120,301).  visible(fomdgt3_6,301). happensAt(walking(fomdgt3_6),301).
orientation(fomdgt3_1,150,302). coords(fomdgt3_1,46,96,302).  visible(fomdgt3_1,302). happensAt(inactive(fomdgt3_1),302). orientation(fomdgt3_6,130,302). coords(fomdgt3_6,215,120,302).  visible(fomdgt3_6,302). happensAt(walking(fomdgt3_6),302).
orientation(fomdgt3_1,150,303). coords(fomdgt3_1,54,100,303).  visible(fomdgt3_1,303). happensAt(inactive(fomdgt3_1),303). orientation(fomdgt3_6,130,303). coords(fomdgt3_6,215,119,303).  visible(fomdgt3_6,303). happensAt(walking(fomdgt3_6),303).
orientation(fomdgt3_1,150,304). coords(fomdgt3_1,54,100,304).  visible(fomdgt3_1,304). happensAt(inactive(fomdgt3_1),304). orientation(fomdgt3_6,130,304). coords(fomdgt3_6,213,115,304).  visible(fomdgt3_6,304). happensAt(walking(fomdgt3_6),304).
orientation(fomdgt3_1,150,305). coords(fomdgt3_1,54,100,305).  visible(fomdgt3_1,305). happensAt(inactive(fomdgt3_1),305). orientation(fomdgt3_6,130,305). coords(fomdgt3_6,213,115,305).  visible(fomdgt3_6,305). happensAt(walking(fomdgt3_6),305).
orientation(fomdgt3_1,150,306). coords(fomdgt3_1,54,100,306).  visible(fomdgt3_1,306). happensAt(inactive(fomdgt3_1),306). orientation(fomdgt3_6,130,306). coords(fomdgt3_6,213,113,306).  visible(fomdgt3_6,306). happensAt(walking(fomdgt3_6),306).
orientation(fomdgt3_1,150,307). coords(fomdgt3_1,54,100,307).  visible(fomdgt3_1,307). happensAt(inactive(fomdgt3_1),307). orientation(fomdgt3_6,130,307). coords(fomdgt3_6,213,113,307).  visible(fomdgt3_6,307). happensAt(walking(fomdgt3_6),307).
orientation(fomdgt3_1,150,308). coords(fomdgt3_1,54,100,308).  visible(fomdgt3_1,308). happensAt(inactive(fomdgt3_1),308). orientation(fomdgt3_6,130,308). coords(fomdgt3_6,212,111,308).  visible(fomdgt3_6,308). happensAt(walking(fomdgt3_6),308).
orientation(fomdgt3_1,150,309). coords(fomdgt3_1,54,100,309).  visible(fomdgt3_1,309). happensAt(inactive(fomdgt3_1),309). orientation(fomdgt3_6,130,309). coords(fomdgt3_6,212,111,309).  visible(fomdgt3_6,309). happensAt(walking(fomdgt3_6),309).
orientation(fomdgt3_1,150,310). coords(fomdgt3_1,54,100,310).  visible(fomdgt3_1,310). happensAt(inactive(fomdgt3_1),310). orientation(fomdgt3_6,130,310). coords(fomdgt3_6,212,112,310).  visible(fomdgt3_6,310). happensAt(walking(fomdgt3_6),310).
orientation(fomdgt3_1,150,311). coords(fomdgt3_1,54,100,311).  visible(fomdgt3_1,311). happensAt(inactive(fomdgt3_1),311). orientation(fomdgt3_6,130,311). coords(fomdgt3_6,212,112,311).  visible(fomdgt3_6,311). happensAt(walking(fomdgt3_6),311).
orientation(fomdgt3_1,150,312). coords(fomdgt3_1,54,100,312).  visible(fomdgt3_1,312). happensAt(inactive(fomdgt3_1),312). orientation(fomdgt3_6,130,312). coords(fomdgt3_6,212,112,312).  visible(fomdgt3_6,312). happensAt(walking(fomdgt3_6),312).
orientation(fomdgt3_1,150,313). coords(fomdgt3_1,54,100,313).  visible(fomdgt3_1,313). happensAt(inactive(fomdgt3_1),313). orientation(fomdgt3_6,130,313). coords(fomdgt3_6,212,111,313).  visible(fomdgt3_6,313). happensAt(walking(fomdgt3_6),313).
orientation(fomdgt3_1,150,314). coords(fomdgt3_1,54,100,314).  visible(fomdgt3_1,314). happensAt(inactive(fomdgt3_1),314). orientation(fomdgt3_6,130,314). coords(fomdgt3_6,212,111,314).  visible(fomdgt3_6,314). happensAt(walking(fomdgt3_6),314).
orientation(fomdgt3_1,150,315). coords(fomdgt3_1,54,100,315).  visible(fomdgt3_1,315). happensAt(inactive(fomdgt3_1),315). orientation(fomdgt3_6,130,315). coords(fomdgt3_6,212,111,315).  visible(fomdgt3_6,315). happensAt(walking(fomdgt3_6),315).
orientation(fomdgt3_1,150,316). coords(fomdgt3_1,54,100,316).  visible(fomdgt3_1,316). happensAt(inactive(fomdgt3_1),316). orientation(fomdgt3_6,130,316). coords(fomdgt3_6,212,110,316).  visible(fomdgt3_6,316). happensAt(walking(fomdgt3_6),316).
orientation(fomdgt3_1,150,317). coords(fomdgt3_1,54,100,317).  visible(fomdgt3_1,317). happensAt(inactive(fomdgt3_1),317). orientation(fomdgt3_6,130,317). coords(fomdgt3_6,212,110,317).  visible(fomdgt3_6,317). happensAt(walking(fomdgt3_6),317).
orientation(fomdgt3_1,150,318). coords(fomdgt3_1,54,100,318).  visible(fomdgt3_1,318). happensAt(inactive(fomdgt3_1),318). orientation(fomdgt3_6,130,318). coords(fomdgt3_6,212,110,318).  visible(fomdgt3_6,318). happensAt(walking(fomdgt3_6),318).
#show holdsAt/2.