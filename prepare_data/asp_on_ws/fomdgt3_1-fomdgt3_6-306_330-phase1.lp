
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
orientation(fomdgt3_1,150,319). coords(fomdgt3_1,54,100,319).  visible(fomdgt3_1,319). happensAt(inactive(fomdgt3_1),319). orientation(fomdgt3_6,130,319). coords(fomdgt3_6,212,110,319).  visible(fomdgt3_6,319). happensAt(walking(fomdgt3_6),319).
orientation(fomdgt3_1,150,320). coords(fomdgt3_1,54,100,320).  visible(fomdgt3_1,320). happensAt(inactive(fomdgt3_1),320). orientation(fomdgt3_6,130,320). coords(fomdgt3_6,212,110,320).  visible(fomdgt3_6,320). happensAt(walking(fomdgt3_6),320).
orientation(fomdgt3_1,150,321). coords(fomdgt3_1,54,100,321).  visible(fomdgt3_1,321). happensAt(inactive(fomdgt3_1),321). orientation(fomdgt3_6,130,321). coords(fomdgt3_6,212,110,321).  visible(fomdgt3_6,321). happensAt(walking(fomdgt3_6),321).
orientation(fomdgt3_1,150,322). coords(fomdgt3_1,54,100,322).  visible(fomdgt3_1,322). happensAt(inactive(fomdgt3_1),322). orientation(fomdgt3_6,130,322). coords(fomdgt3_6,212,112,322).  visible(fomdgt3_6,322). happensAt(walking(fomdgt3_6),322).
orientation(fomdgt3_1,150,323). coords(fomdgt3_1,54,100,323).  visible(fomdgt3_1,323). happensAt(inactive(fomdgt3_1),323). orientation(fomdgt3_6,130,323). coords(fomdgt3_6,212,112,323).  visible(fomdgt3_6,323). happensAt(walking(fomdgt3_6),323).
orientation(fomdgt3_1,150,324). coords(fomdgt3_1,54,100,324).  visible(fomdgt3_1,324). happensAt(inactive(fomdgt3_1),324). orientation(fomdgt3_6,130,324). coords(fomdgt3_6,212,112,324).  visible(fomdgt3_6,324). happensAt(walking(fomdgt3_6),324).
orientation(fomdgt3_1,150,325). coords(fomdgt3_1,54,100,325).  visible(fomdgt3_1,325). happensAt(inactive(fomdgt3_1),325). orientation(fomdgt3_6,130,325). coords(fomdgt3_6,212,112,325).  visible(fomdgt3_6,325). happensAt(walking(fomdgt3_6),325).
orientation(fomdgt3_1,150,326). coords(fomdgt3_1,54,100,326).  visible(fomdgt3_1,326). happensAt(inactive(fomdgt3_1),326). orientation(fomdgt3_6,130,326). coords(fomdgt3_6,212,112,326).  visible(fomdgt3_6,326). happensAt(walking(fomdgt3_6),326).
orientation(fomdgt3_1,150,327). coords(fomdgt3_1,54,100,327).  visible(fomdgt3_1,327). happensAt(inactive(fomdgt3_1),327). orientation(fomdgt3_6,130,327). coords(fomdgt3_6,212,112,327).  visible(fomdgt3_6,327). happensAt(walking(fomdgt3_6),327).
orientation(fomdgt3_1,150,328). coords(fomdgt3_1,54,100,328).  visible(fomdgt3_1,328). happensAt(inactive(fomdgt3_1),328). orientation(fomdgt3_6,130,328). coords(fomdgt3_6,212,111,328).  visible(fomdgt3_6,328). happensAt(walking(fomdgt3_6),328).
orientation(fomdgt3_1,150,329). coords(fomdgt3_1,54,100,329).  visible(fomdgt3_1,329). happensAt(inactive(fomdgt3_1),329). orientation(fomdgt3_6,130,329). coords(fomdgt3_6,212,111,329).  visible(fomdgt3_6,329). happensAt(walking(fomdgt3_6),329).
orientation(fomdgt3_1,150,330). coords(fomdgt3_1,54,100,330).  visible(fomdgt3_1,330). happensAt(inactive(fomdgt3_1),330). orientation(fomdgt3_6,130,330). coords(fomdgt3_6,212,111,330).  visible(fomdgt3_6,330). happensAt(walking(fomdgt3_6),330).
#show holdsAt/2.