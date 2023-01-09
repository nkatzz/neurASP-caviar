
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

	orientation(bww1gt_1,3,302). coords(bww1gt_1,104,264,302).  visible(bww1gt_1,302). happensAt(inactive(bww1gt_1),302). orientation(bww1gt_2,167,302). coords(bww1gt_2,57,162,302).  visible(bww1gt_2,302). happensAt(walking(bww1gt_2),302).
orientation(bww1gt_1,176,303). coords(bww1gt_1,104,264,303).  visible(bww1gt_1,303). happensAt(inactive(bww1gt_1),303). orientation(bww1gt_2,167,303). coords(bww1gt_2,58,162,303).  visible(bww1gt_2,303). happensAt(walking(bww1gt_2),303).
orientation(bww1gt_1,175,304). coords(bww1gt_1,104,264,304).  visible(bww1gt_1,304). happensAt(inactive(bww1gt_1),304). orientation(bww1gt_2,167,304). coords(bww1gt_2,58,162,304).  visible(bww1gt_2,304). happensAt(walking(bww1gt_2),304).
orientation(bww1gt_1,177,305). coords(bww1gt_1,104,264,305).  visible(bww1gt_1,305). happensAt(inactive(bww1gt_1),305). orientation(bww1gt_2,167,305). coords(bww1gt_2,59,162,305).  visible(bww1gt_2,305). happensAt(walking(bww1gt_2),305).
orientation(bww1gt_1,179,306). coords(bww1gt_1,104,264,306).  visible(bww1gt_1,306). happensAt(inactive(bww1gt_1),306). orientation(bww1gt_2,167,306). coords(bww1gt_2,59,162,306).  visible(bww1gt_2,306). happensAt(walking(bww1gt_2),306).
orientation(bww1gt_1,2,307). coords(bww1gt_1,104,264,307).  visible(bww1gt_1,307). happensAt(inactive(bww1gt_1),307). orientation(bww1gt_2,167,307). coords(bww1gt_2,60,161,307).  visible(bww1gt_2,307). happensAt(walking(bww1gt_2),307).
orientation(bww1gt_1,2,308). coords(bww1gt_1,104,264,308).  visible(bww1gt_1,308). happensAt(inactive(bww1gt_1),308). orientation(bww1gt_2,167,308). coords(bww1gt_2,61,161,308).  visible(bww1gt_2,308). happensAt(walking(bww1gt_2),308).
orientation(bww1gt_1,1,309). coords(bww1gt_1,104,264,309).  visible(bww1gt_1,309). happensAt(inactive(bww1gt_1),309). orientation(bww1gt_2,167,309). coords(bww1gt_2,62,161,309).  visible(bww1gt_2,309). happensAt(walking(bww1gt_2),309).
orientation(bww1gt_1,3,310). coords(bww1gt_1,104,264,310).  visible(bww1gt_1,310). happensAt(inactive(bww1gt_1),310). orientation(bww1gt_2,167,310). coords(bww1gt_2,62,161,310).  visible(bww1gt_2,310). happensAt(walking(bww1gt_2),310).
orientation(bww1gt_1,2,311). coords(bww1gt_1,104,264,311).  visible(bww1gt_1,311). happensAt(inactive(bww1gt_1),311). orientation(bww1gt_2,167,311). coords(bww1gt_2,62,161,311).  visible(bww1gt_2,311). happensAt(walking(bww1gt_2),311).
orientation(bww1gt_1,6,312). coords(bww1gt_1,105,264,312).  visible(bww1gt_1,312). happensAt(inactive(bww1gt_1),312). orientation(bww1gt_2,167,312). coords(bww1gt_2,62,161,312).  visible(bww1gt_2,312). happensAt(walking(bww1gt_2),312).
orientation(bww1gt_1,5,313). coords(bww1gt_1,105,264,313).  visible(bww1gt_1,313). happensAt(inactive(bww1gt_1),313). orientation(bww1gt_2,167,313). coords(bww1gt_2,63,161,313).  visible(bww1gt_2,313). happensAt(walking(bww1gt_2),313).
orientation(bww1gt_1,7,314). coords(bww1gt_1,105,264,314).  visible(bww1gt_1,314). happensAt(inactive(bww1gt_1),314). orientation(bww1gt_2,167,314). coords(bww1gt_2,63,161,314).  visible(bww1gt_2,314). happensAt(walking(bww1gt_2),314).
orientation(bww1gt_1,4,315). coords(bww1gt_1,105,264,315).  visible(bww1gt_1,315). happensAt(inactive(bww1gt_1),315). orientation(bww1gt_2,167,315). coords(bww1gt_2,63,161,315).  visible(bww1gt_2,315). happensAt(walking(bww1gt_2),315).
orientation(bww1gt_1,6,316). coords(bww1gt_1,105,264,316).  visible(bww1gt_1,316). happensAt(inactive(bww1gt_1),316). orientation(bww1gt_2,167,316). coords(bww1gt_2,63,161,316).  visible(bww1gt_2,316). happensAt(walking(bww1gt_2),316).
orientation(bww1gt_1,4,317). coords(bww1gt_1,105,264,317).  visible(bww1gt_1,317). happensAt(inactive(bww1gt_1),317). orientation(bww1gt_2,167,317). coords(bww1gt_2,63,161,317).  visible(bww1gt_2,317). happensAt(walking(bww1gt_2),317).
orientation(bww1gt_1,4,318). coords(bww1gt_1,105,264,318).  visible(bww1gt_1,318). happensAt(inactive(bww1gt_1),318). orientation(bww1gt_2,167,318). coords(bww1gt_2,63,161,318).  visible(bww1gt_2,318). happensAt(walking(bww1gt_2),318).
orientation(bww1gt_1,2,319). coords(bww1gt_1,105,264,319).  visible(bww1gt_1,319). happensAt(inactive(bww1gt_1),319). orientation(bww1gt_2,167,319). coords(bww1gt_2,64,161,319).  visible(bww1gt_2,319). happensAt(walking(bww1gt_2),319).
orientation(bww1gt_1,4,320). coords(bww1gt_1,105,264,320).  visible(bww1gt_1,320). happensAt(inactive(bww1gt_1),320). orientation(bww1gt_2,167,320). coords(bww1gt_2,65,161,320).  visible(bww1gt_2,320). happensAt(walking(bww1gt_2),320).
orientation(bww1gt_1,4,321). coords(bww1gt_1,105,264,321).  visible(bww1gt_1,321). happensAt(inactive(bww1gt_1),321). orientation(bww1gt_2,167,321). coords(bww1gt_2,65,161,321).  visible(bww1gt_2,321). happensAt(walking(bww1gt_2),321).
orientation(bww1gt_1,5,322). coords(bww1gt_1,105,264,322).  visible(bww1gt_1,322). happensAt(inactive(bww1gt_1),322). orientation(bww1gt_2,167,322). coords(bww1gt_2,66,161,322).  visible(bww1gt_2,322). happensAt(walking(bww1gt_2),322).
orientation(bww1gt_1,8,323). coords(bww1gt_1,105,264,323).  visible(bww1gt_1,323). happensAt(inactive(bww1gt_1),323). orientation(bww1gt_2,167,323). coords(bww1gt_2,67,161,323).  visible(bww1gt_2,323). happensAt(walking(bww1gt_2),323).
orientation(bww1gt_1,8,324). coords(bww1gt_1,105,264,324).  visible(bww1gt_1,324). happensAt(inactive(bww1gt_1),324). orientation(bww1gt_2,167,324). coords(bww1gt_2,68,160,324).  visible(bww1gt_2,324). happensAt(walking(bww1gt_2),324).
orientation(bww1gt_1,7,325). coords(bww1gt_1,105,264,325).  visible(bww1gt_1,325). happensAt(inactive(bww1gt_1),325). orientation(bww1gt_2,167,325). coords(bww1gt_2,68,160,325).  visible(bww1gt_2,325). happensAt(walking(bww1gt_2),325).
orientation(bww1gt_1,179,326). coords(bww1gt_1,105,264,326).  visible(bww1gt_1,326). happensAt(inactive(bww1gt_1),326). orientation(bww1gt_2,167,326). coords(bww1gt_2,69,160,326).  visible(bww1gt_2,326). happensAt(walking(bww1gt_2),326).
#show holdsAt/2.