
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

	orientation(fomdgt3_2,130,294). coords(fomdgt3_2,223,121,294).  visible(fomdgt3_2,294). happensAt(walking(fomdgt3_2),294). orientation(fomdgt3_6,130,294). coords(fomdgt3_6,217,133,294).  visible(fomdgt3_6,294). happensAt(walking(fomdgt3_6),294).
orientation(fomdgt3_2,130,295). coords(fomdgt3_2,223,119,295).  visible(fomdgt3_2,295). happensAt(walking(fomdgt3_2),295). orientation(fomdgt3_6,130,295). coords(fomdgt3_6,217,130,295).  visible(fomdgt3_6,295). happensAt(walking(fomdgt3_6),295).
orientation(fomdgt3_2,130,296). coords(fomdgt3_2,223,120,296).  visible(fomdgt3_2,296). happensAt(walking(fomdgt3_2),296). orientation(fomdgt3_6,130,296). coords(fomdgt3_6,217,127,296).  visible(fomdgt3_6,296). happensAt(walking(fomdgt3_6),296).
orientation(fomdgt3_2,110,297). coords(fomdgt3_2,224,120,297).  visible(fomdgt3_2,297). happensAt(walking(fomdgt3_2),297). orientation(fomdgt3_6,130,297). coords(fomdgt3_6,217,125,297).  visible(fomdgt3_6,297). happensAt(walking(fomdgt3_6),297).
orientation(fomdgt3_2,110,298). coords(fomdgt3_2,224,121,298).  visible(fomdgt3_2,298). happensAt(walking(fomdgt3_2),298). orientation(fomdgt3_6,130,298). coords(fomdgt3_6,217,124,298).  visible(fomdgt3_6,298). happensAt(walking(fomdgt3_6),298).
orientation(fomdgt3_2,110,299). coords(fomdgt3_2,223,121,299).  visible(fomdgt3_2,299). happensAt(walking(fomdgt3_2),299). orientation(fomdgt3_6,130,299). coords(fomdgt3_6,217,123,299).  visible(fomdgt3_6,299). happensAt(walking(fomdgt3_6),299).
orientation(fomdgt3_2,110,300). coords(fomdgt3_2,223,121,300).  visible(fomdgt3_2,300). happensAt(walking(fomdgt3_2),300). orientation(fomdgt3_6,130,300). coords(fomdgt3_6,217,123,300).  visible(fomdgt3_6,300). happensAt(walking(fomdgt3_6),300).
orientation(fomdgt3_2,120,301). coords(fomdgt3_2,230,115,301).  visible(fomdgt3_2,301). happensAt(walking(fomdgt3_2),301). orientation(fomdgt3_6,130,301). coords(fomdgt3_6,216,120,301).  visible(fomdgt3_6,301). happensAt(walking(fomdgt3_6),301).
orientation(fomdgt3_2,120,302). coords(fomdgt3_2,230,115,302).  visible(fomdgt3_2,302). happensAt(walking(fomdgt3_2),302). orientation(fomdgt3_6,130,302). coords(fomdgt3_6,215,120,302).  visible(fomdgt3_6,302). happensAt(walking(fomdgt3_6),302).
orientation(fomdgt3_2,120,303). coords(fomdgt3_2,230,115,303).  visible(fomdgt3_2,303). happensAt(walking(fomdgt3_2),303). orientation(fomdgt3_6,130,303). coords(fomdgt3_6,215,119,303).  visible(fomdgt3_6,303). happensAt(walking(fomdgt3_6),303).
orientation(fomdgt3_2,120,304). coords(fomdgt3_2,232,115,304).  visible(fomdgt3_2,304). happensAt(walking(fomdgt3_2),304). orientation(fomdgt3_6,130,304). coords(fomdgt3_6,213,115,304).  visible(fomdgt3_6,304). happensAt(walking(fomdgt3_6),304).
orientation(fomdgt3_2,120,305). coords(fomdgt3_2,232,115,305).  visible(fomdgt3_2,305). happensAt(walking(fomdgt3_2),305). orientation(fomdgt3_6,130,305). coords(fomdgt3_6,213,115,305).  visible(fomdgt3_6,305). happensAt(walking(fomdgt3_6),305).
orientation(fomdgt3_2,120,306). coords(fomdgt3_2,233,116,306).  visible(fomdgt3_2,306). happensAt(walking(fomdgt3_2),306). orientation(fomdgt3_6,130,306). coords(fomdgt3_6,213,113,306).  visible(fomdgt3_6,306). happensAt(walking(fomdgt3_6),306).
orientation(fomdgt3_2,120,307). coords(fomdgt3_2,233,116,307).  visible(fomdgt3_2,307). happensAt(walking(fomdgt3_2),307). orientation(fomdgt3_6,130,307). coords(fomdgt3_6,213,113,307).  visible(fomdgt3_6,307). happensAt(walking(fomdgt3_6),307).
orientation(fomdgt3_2,120,308). coords(fomdgt3_2,239,116,308).  visible(fomdgt3_2,308). happensAt(walking(fomdgt3_2),308). orientation(fomdgt3_6,130,308). coords(fomdgt3_6,212,111,308).  visible(fomdgt3_6,308). happensAt(walking(fomdgt3_6),308).
orientation(fomdgt3_2,120,309). coords(fomdgt3_2,240,116,309).  visible(fomdgt3_2,309). happensAt(walking(fomdgt3_2),309). orientation(fomdgt3_6,130,309). coords(fomdgt3_6,212,111,309).  visible(fomdgt3_6,309). happensAt(walking(fomdgt3_6),309).
orientation(fomdgt3_2,120,310). coords(fomdgt3_2,240,116,310).  visible(fomdgt3_2,310). happensAt(walking(fomdgt3_2),310). orientation(fomdgt3_6,130,310). coords(fomdgt3_6,212,112,310).  visible(fomdgt3_6,310). happensAt(walking(fomdgt3_6),310).
orientation(fomdgt3_2,120,311). coords(fomdgt3_2,240,116,311).  visible(fomdgt3_2,311). happensAt(walking(fomdgt3_2),311). orientation(fomdgt3_6,130,311). coords(fomdgt3_6,212,112,311).  visible(fomdgt3_6,311). happensAt(walking(fomdgt3_6),311).
orientation(fomdgt3_2,120,312). coords(fomdgt3_2,240,116,312).  visible(fomdgt3_2,312). happensAt(walking(fomdgt3_2),312). orientation(fomdgt3_6,130,312). coords(fomdgt3_6,212,112,312).  visible(fomdgt3_6,312). happensAt(walking(fomdgt3_6),312).
orientation(fomdgt3_2,120,313). coords(fomdgt3_2,241,115,313).  visible(fomdgt3_2,313). happensAt(walking(fomdgt3_2),313). orientation(fomdgt3_6,130,313). coords(fomdgt3_6,212,111,313).  visible(fomdgt3_6,313). happensAt(walking(fomdgt3_6),313).
orientation(fomdgt3_2,120,314). coords(fomdgt3_2,242,114,314).  visible(fomdgt3_2,314). happensAt(walking(fomdgt3_2),314). orientation(fomdgt3_6,130,314). coords(fomdgt3_6,212,111,314).  visible(fomdgt3_6,314). happensAt(walking(fomdgt3_6),314).
orientation(fomdgt3_2,120,315). coords(fomdgt3_2,243,113,315).  visible(fomdgt3_2,315). happensAt(walking(fomdgt3_2),315). orientation(fomdgt3_6,130,315). coords(fomdgt3_6,212,111,315).  visible(fomdgt3_6,315). happensAt(walking(fomdgt3_6),315).
orientation(fomdgt3_2,120,316). coords(fomdgt3_2,243,111,316).  visible(fomdgt3_2,316). happensAt(walking(fomdgt3_2),316). orientation(fomdgt3_6,130,316). coords(fomdgt3_6,212,110,316).  visible(fomdgt3_6,316). happensAt(walking(fomdgt3_6),316).
orientation(fomdgt3_2,80,317). coords(fomdgt3_2,243,110,317).  visible(fomdgt3_2,317). happensAt(walking(fomdgt3_2),317). orientation(fomdgt3_6,130,317). coords(fomdgt3_6,212,110,317).  visible(fomdgt3_6,317). happensAt(walking(fomdgt3_6),317).
orientation(fomdgt3_2,80,318). coords(fomdgt3_2,244,110,318).  visible(fomdgt3_2,318). happensAt(walking(fomdgt3_2),318). orientation(fomdgt3_6,130,318). coords(fomdgt3_6,212,110,318).  visible(fomdgt3_6,318). happensAt(walking(fomdgt3_6),318).
#show holdsAt/2.