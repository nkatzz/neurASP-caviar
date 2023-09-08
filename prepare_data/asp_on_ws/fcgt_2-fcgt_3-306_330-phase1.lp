
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

	orientation(fcgt_2,150,306). coords(fcgt_2,92,129,306).  visible(fcgt_2,306). happensAt(active(fcgt_2),306). orientation(fcgt_3,164,306). coords(fcgt_3,98,187,306).  visible(fcgt_3,306). happensAt(inactive(fcgt_3),306).
orientation(fcgt_2,150,307). coords(fcgt_2,92,129,307).  visible(fcgt_2,307). happensAt(active(fcgt_2),307). orientation(fcgt_3,164,307). coords(fcgt_3,98,187,307).  visible(fcgt_3,307). happensAt(inactive(fcgt_3),307).
orientation(fcgt_2,150,308). coords(fcgt_2,92,129,308).  visible(fcgt_2,308). happensAt(active(fcgt_2),308). orientation(fcgt_3,164,308). coords(fcgt_3,98,187,308).  visible(fcgt_3,308). happensAt(inactive(fcgt_3),308).
orientation(fcgt_2,150,309). coords(fcgt_2,92,129,309).  visible(fcgt_2,309). happensAt(active(fcgt_2),309). orientation(fcgt_3,164,309). coords(fcgt_3,98,187,309).  visible(fcgt_3,309). happensAt(inactive(fcgt_3),309).
orientation(fcgt_2,150,310). coords(fcgt_2,93,129,310).  visible(fcgt_2,310). happensAt(active(fcgt_2),310). orientation(fcgt_3,164,310). coords(fcgt_3,98,187,310).  visible(fcgt_3,310). happensAt(inactive(fcgt_3),310).
orientation(fcgt_2,150,311). coords(fcgt_2,94,129,311).  visible(fcgt_2,311). happensAt(active(fcgt_2),311). orientation(fcgt_3,164,311). coords(fcgt_3,98,187,311).  visible(fcgt_3,311). happensAt(inactive(fcgt_3),311).
orientation(fcgt_2,150,312). coords(fcgt_2,93,129,312).  visible(fcgt_2,312). happensAt(active(fcgt_2),312). orientation(fcgt_3,164,312). coords(fcgt_3,98,188,312).  visible(fcgt_3,312). happensAt(inactive(fcgt_3),312).
orientation(fcgt_2,150,313). coords(fcgt_2,95,129,313).  visible(fcgt_2,313). happensAt(active(fcgt_2),313). orientation(fcgt_3,164,313). coords(fcgt_3,98,188,313).  visible(fcgt_3,313). happensAt(inactive(fcgt_3),313).
orientation(fcgt_2,150,314). coords(fcgt_2,95,129,314).  visible(fcgt_2,314). happensAt(active(fcgt_2),314). orientation(fcgt_3,164,314). coords(fcgt_3,98,188,314).  visible(fcgt_3,314). happensAt(inactive(fcgt_3),314).
orientation(fcgt_2,150,315). coords(fcgt_2,95,129,315).  visible(fcgt_2,315). happensAt(active(fcgt_2),315). orientation(fcgt_3,164,315). coords(fcgt_3,98,188,315).  visible(fcgt_3,315). happensAt(inactive(fcgt_3),315).
orientation(fcgt_2,150,316). coords(fcgt_2,95,129,316).  visible(fcgt_2,316). happensAt(active(fcgt_2),316). orientation(fcgt_3,164,316). coords(fcgt_3,98,188,316).  visible(fcgt_3,316). happensAt(inactive(fcgt_3),316).
orientation(fcgt_2,150,317). coords(fcgt_2,95,129,317).  visible(fcgt_2,317). happensAt(active(fcgt_2),317). orientation(fcgt_3,164,317). coords(fcgt_3,98,188,317).  visible(fcgt_3,317). happensAt(inactive(fcgt_3),317).
orientation(fcgt_2,150,318). coords(fcgt_2,95,129,318).  visible(fcgt_2,318). happensAt(active(fcgt_2),318). orientation(fcgt_3,164,318). coords(fcgt_3,99,188,318).  visible(fcgt_3,318). happensAt(inactive(fcgt_3),318).
orientation(fcgt_2,150,319). coords(fcgt_2,94,129,319).  visible(fcgt_2,319). happensAt(active(fcgt_2),319). orientation(fcgt_3,164,319). coords(fcgt_3,98,187,319).  visible(fcgt_3,319). happensAt(inactive(fcgt_3),319).
orientation(fcgt_2,150,320). coords(fcgt_2,94,129,320).  visible(fcgt_2,320). happensAt(active(fcgt_2),320). orientation(fcgt_3,164,320). coords(fcgt_3,99,187,320).  visible(fcgt_3,320). happensAt(inactive(fcgt_3),320).
orientation(fcgt_2,150,321). coords(fcgt_2,94,129,321).  visible(fcgt_2,321). happensAt(active(fcgt_2),321). orientation(fcgt_3,164,321). coords(fcgt_3,98,187,321).  visible(fcgt_3,321). happensAt(inactive(fcgt_3),321).
orientation(fcgt_2,150,322). coords(fcgt_2,94,129,322).  visible(fcgt_2,322). happensAt(active(fcgt_2),322). orientation(fcgt_3,164,322). coords(fcgt_3,98,187,322).  visible(fcgt_3,322). happensAt(inactive(fcgt_3),322).
orientation(fcgt_2,150,323). coords(fcgt_2,94,129,323).  visible(fcgt_2,323). happensAt(active(fcgt_2),323). orientation(fcgt_3,164,323). coords(fcgt_3,98,187,323).  visible(fcgt_3,323). happensAt(inactive(fcgt_3),323).
orientation(fcgt_2,150,324). coords(fcgt_2,94,129,324).  visible(fcgt_2,324). happensAt(active(fcgt_2),324). orientation(fcgt_3,164,324). coords(fcgt_3,98,187,324).  visible(fcgt_3,324). happensAt(inactive(fcgt_3),324).
orientation(fcgt_2,150,325). coords(fcgt_2,93,128,325).  visible(fcgt_2,325). happensAt(active(fcgt_2),325). orientation(fcgt_3,164,325). coords(fcgt_3,98,187,325).  visible(fcgt_3,325). happensAt(inactive(fcgt_3),325).
orientation(fcgt_2,146,326). coords(fcgt_2,93,128,326).  visible(fcgt_2,326). happensAt(active(fcgt_2),326). orientation(fcgt_3,164,326). coords(fcgt_3,98,187,326).  visible(fcgt_3,326). happensAt(inactive(fcgt_3),326).
orientation(fcgt_2,146,327). coords(fcgt_2,92,127,327).  visible(fcgt_2,327). happensAt(active(fcgt_2),327). orientation(fcgt_3,164,327). coords(fcgt_3,98,187,327).  visible(fcgt_3,327). happensAt(inactive(fcgt_3),327).
orientation(fcgt_2,146,328). coords(fcgt_2,91,127,328).  visible(fcgt_2,328). happensAt(active(fcgt_2),328). orientation(fcgt_3,164,328). coords(fcgt_3,98,187,328).  visible(fcgt_3,328). happensAt(inactive(fcgt_3),328).
orientation(fcgt_2,146,329). coords(fcgt_2,91,126,329).  visible(fcgt_2,329). happensAt(active(fcgt_2),329). orientation(fcgt_3,164,329). coords(fcgt_3,98,187,329).  visible(fcgt_3,329). happensAt(inactive(fcgt_3),329).
orientation(fcgt_2,146,330). coords(fcgt_2,91,125,330).  visible(fcgt_2,330). happensAt(active(fcgt_2),330). orientation(fcgt_3,164,330). coords(fcgt_3,98,187,330).  visible(fcgt_3,330). happensAt(inactive(fcgt_3),330).
#show holdsAt/2.