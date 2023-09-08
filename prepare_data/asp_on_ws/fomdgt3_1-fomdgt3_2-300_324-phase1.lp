
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

	orientation(fomdgt3_1,150,300). coords(fomdgt3_1,46,96,300).  visible(fomdgt3_1,300). happensAt(inactive(fomdgt3_1),300). orientation(fomdgt3_2,110,300). coords(fomdgt3_2,223,121,300).  visible(fomdgt3_2,300). happensAt(walking(fomdgt3_2),300).
orientation(fomdgt3_1,150,301). coords(fomdgt3_1,46,96,301).  visible(fomdgt3_1,301). happensAt(inactive(fomdgt3_1),301). orientation(fomdgt3_2,120,301). coords(fomdgt3_2,230,115,301).  visible(fomdgt3_2,301). happensAt(walking(fomdgt3_2),301).
orientation(fomdgt3_1,150,302). coords(fomdgt3_1,46,96,302).  visible(fomdgt3_1,302). happensAt(inactive(fomdgt3_1),302). orientation(fomdgt3_2,120,302). coords(fomdgt3_2,230,115,302).  visible(fomdgt3_2,302). happensAt(walking(fomdgt3_2),302).
orientation(fomdgt3_1,150,303). coords(fomdgt3_1,54,100,303).  visible(fomdgt3_1,303). happensAt(inactive(fomdgt3_1),303). orientation(fomdgt3_2,120,303). coords(fomdgt3_2,230,115,303).  visible(fomdgt3_2,303). happensAt(walking(fomdgt3_2),303).
orientation(fomdgt3_1,150,304). coords(fomdgt3_1,54,100,304).  visible(fomdgt3_1,304). happensAt(inactive(fomdgt3_1),304). orientation(fomdgt3_2,120,304). coords(fomdgt3_2,232,115,304).  visible(fomdgt3_2,304). happensAt(walking(fomdgt3_2),304).
orientation(fomdgt3_1,150,305). coords(fomdgt3_1,54,100,305).  visible(fomdgt3_1,305). happensAt(inactive(fomdgt3_1),305). orientation(fomdgt3_2,120,305). coords(fomdgt3_2,232,115,305).  visible(fomdgt3_2,305). happensAt(walking(fomdgt3_2),305).
orientation(fomdgt3_1,150,306). coords(fomdgt3_1,54,100,306).  visible(fomdgt3_1,306). happensAt(inactive(fomdgt3_1),306). orientation(fomdgt3_2,120,306). coords(fomdgt3_2,233,116,306).  visible(fomdgt3_2,306). happensAt(walking(fomdgt3_2),306).
orientation(fomdgt3_1,150,307). coords(fomdgt3_1,54,100,307).  visible(fomdgt3_1,307). happensAt(inactive(fomdgt3_1),307). orientation(fomdgt3_2,120,307). coords(fomdgt3_2,233,116,307).  visible(fomdgt3_2,307). happensAt(walking(fomdgt3_2),307).
orientation(fomdgt3_1,150,308). coords(fomdgt3_1,54,100,308).  visible(fomdgt3_1,308). happensAt(inactive(fomdgt3_1),308). orientation(fomdgt3_2,120,308). coords(fomdgt3_2,239,116,308).  visible(fomdgt3_2,308). happensAt(walking(fomdgt3_2),308).
orientation(fomdgt3_1,150,309). coords(fomdgt3_1,54,100,309).  visible(fomdgt3_1,309). happensAt(inactive(fomdgt3_1),309). orientation(fomdgt3_2,120,309). coords(fomdgt3_2,240,116,309).  visible(fomdgt3_2,309). happensAt(walking(fomdgt3_2),309).
orientation(fomdgt3_1,150,310). coords(fomdgt3_1,54,100,310).  visible(fomdgt3_1,310). happensAt(inactive(fomdgt3_1),310). orientation(fomdgt3_2,120,310). coords(fomdgt3_2,240,116,310).  visible(fomdgt3_2,310). happensAt(walking(fomdgt3_2),310).
orientation(fomdgt3_1,150,311). coords(fomdgt3_1,54,100,311).  visible(fomdgt3_1,311). happensAt(inactive(fomdgt3_1),311). orientation(fomdgt3_2,120,311). coords(fomdgt3_2,240,116,311).  visible(fomdgt3_2,311). happensAt(walking(fomdgt3_2),311).
orientation(fomdgt3_1,150,312). coords(fomdgt3_1,54,100,312).  visible(fomdgt3_1,312). happensAt(inactive(fomdgt3_1),312). orientation(fomdgt3_2,120,312). coords(fomdgt3_2,240,116,312).  visible(fomdgt3_2,312). happensAt(walking(fomdgt3_2),312).
orientation(fomdgt3_1,150,313). coords(fomdgt3_1,54,100,313).  visible(fomdgt3_1,313). happensAt(inactive(fomdgt3_1),313). orientation(fomdgt3_2,120,313). coords(fomdgt3_2,241,115,313).  visible(fomdgt3_2,313). happensAt(walking(fomdgt3_2),313).
orientation(fomdgt3_1,150,314). coords(fomdgt3_1,54,100,314).  visible(fomdgt3_1,314). happensAt(inactive(fomdgt3_1),314). orientation(fomdgt3_2,120,314). coords(fomdgt3_2,242,114,314).  visible(fomdgt3_2,314). happensAt(walking(fomdgt3_2),314).
orientation(fomdgt3_1,150,315). coords(fomdgt3_1,54,100,315).  visible(fomdgt3_1,315). happensAt(inactive(fomdgt3_1),315). orientation(fomdgt3_2,120,315). coords(fomdgt3_2,243,113,315).  visible(fomdgt3_2,315). happensAt(walking(fomdgt3_2),315).
orientation(fomdgt3_1,150,316). coords(fomdgt3_1,54,100,316).  visible(fomdgt3_1,316). happensAt(inactive(fomdgt3_1),316). orientation(fomdgt3_2,120,316). coords(fomdgt3_2,243,111,316).  visible(fomdgt3_2,316). happensAt(walking(fomdgt3_2),316).
orientation(fomdgt3_1,150,317). coords(fomdgt3_1,54,100,317).  visible(fomdgt3_1,317). happensAt(inactive(fomdgt3_1),317). orientation(fomdgt3_2,80,317). coords(fomdgt3_2,243,110,317).  visible(fomdgt3_2,317). happensAt(walking(fomdgt3_2),317).
orientation(fomdgt3_1,150,318). coords(fomdgt3_1,54,100,318).  visible(fomdgt3_1,318). happensAt(inactive(fomdgt3_1),318). orientation(fomdgt3_2,80,318). coords(fomdgt3_2,244,110,318).  visible(fomdgt3_2,318). happensAt(walking(fomdgt3_2),318).
orientation(fomdgt3_1,150,319). coords(fomdgt3_1,54,100,319).  visible(fomdgt3_1,319). happensAt(inactive(fomdgt3_1),319). orientation(fomdgt3_2,80,319). coords(fomdgt3_2,249,107,319).  visible(fomdgt3_2,319). happensAt(walking(fomdgt3_2),319).
orientation(fomdgt3_1,150,320). coords(fomdgt3_1,54,100,320).  visible(fomdgt3_1,320). happensAt(inactive(fomdgt3_1),320). orientation(fomdgt3_2,80,320). coords(fomdgt3_2,249,104,320).  visible(fomdgt3_2,320). happensAt(walking(fomdgt3_2),320).
orientation(fomdgt3_1,150,321). coords(fomdgt3_1,54,100,321).  visible(fomdgt3_1,321). happensAt(inactive(fomdgt3_1),321). orientation(fomdgt3_2,80,321). coords(fomdgt3_2,250,104,321).  visible(fomdgt3_2,321). happensAt(walking(fomdgt3_2),321).
orientation(fomdgt3_1,150,322). coords(fomdgt3_1,54,100,322).  visible(fomdgt3_1,322). happensAt(inactive(fomdgt3_1),322). orientation(fomdgt3_2,70,322). coords(fomdgt3_2,248,102,322).  visible(fomdgt3_2,322). happensAt(walking(fomdgt3_2),322).
orientation(fomdgt3_1,150,323). coords(fomdgt3_1,54,100,323).  visible(fomdgt3_1,323). happensAt(inactive(fomdgt3_1),323). orientation(fomdgt3_2,70,323). coords(fomdgt3_2,248,102,323).  visible(fomdgt3_2,323). happensAt(walking(fomdgt3_2),323).
orientation(fomdgt3_1,150,324). coords(fomdgt3_1,54,100,324).  visible(fomdgt3_1,324). happensAt(inactive(fomdgt3_1),324). orientation(fomdgt3_2,60,324). coords(fomdgt3_2,248,98,324).  visible(fomdgt3_2,324). happensAt(walking(fomdgt3_2),324).
#show holdsAt/2.