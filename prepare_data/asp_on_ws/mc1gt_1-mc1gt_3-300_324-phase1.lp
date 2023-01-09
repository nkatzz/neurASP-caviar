
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

	orientation(mc1gt_1,1,300). coords(mc1gt_1,222,245,300).  visible(mc1gt_1,300). happensAt(walking(mc1gt_1),300). orientation(mc1gt_3,135,300). coords(mc1gt_3,283,177,300).  visible(mc1gt_3,300). happensAt(walking(mc1gt_3),300).
orientation(mc1gt_1,5,301). coords(mc1gt_1,223,248,301).  visible(mc1gt_1,301). happensAt(walking(mc1gt_1),301). orientation(mc1gt_3,135,301). coords(mc1gt_3,284,179,301).  visible(mc1gt_3,301). happensAt(walking(mc1gt_3),301).
orientation(mc1gt_1,14,302). coords(mc1gt_1,223,252,302).  visible(mc1gt_1,302). happensAt(walking(mc1gt_1),302). orientation(mc1gt_3,135,302). coords(mc1gt_3,285,181,302).  visible(mc1gt_3,302). happensAt(walking(mc1gt_3),302).
orientation(mc1gt_1,5,303). coords(mc1gt_1,223,255,303).  visible(mc1gt_1,303). happensAt(walking(mc1gt_1),303). orientation(mc1gt_3,135,303). coords(mc1gt_3,286,182,303).  visible(mc1gt_3,303). happensAt(walking(mc1gt_3),303).
orientation(mc1gt_1,7,304). coords(mc1gt_1,224,259,304).  visible(mc1gt_1,304). happensAt(walking(mc1gt_1),304). orientation(mc1gt_3,135,304). coords(mc1gt_3,287,184,304).  visible(mc1gt_3,304). happensAt(walking(mc1gt_3),304).
orientation(mc1gt_1,13,305). coords(mc1gt_1,224,262,305).  visible(mc1gt_1,305). happensAt(walking(mc1gt_1),305). orientation(mc1gt_3,135,305). coords(mc1gt_3,287,185,305).  visible(mc1gt_3,305). happensAt(walking(mc1gt_3),305).
orientation(mc1gt_1,12,306). coords(mc1gt_1,224,263,306).  visible(mc1gt_1,306). happensAt(walking(mc1gt_1),306). orientation(mc1gt_3,152,306). coords(mc1gt_3,291,188,306).  visible(mc1gt_3,306). happensAt(walking(mc1gt_3),306).
orientation(mc1gt_1,8,307). coords(mc1gt_1,225,264,307).  visible(mc1gt_1,307). happensAt(walking(mc1gt_1),307). orientation(mc1gt_3,160,307). coords(mc1gt_3,293,192,307).  visible(mc1gt_3,307). happensAt(walking(mc1gt_3),307).
orientation(mc1gt_1,15,308). coords(mc1gt_1,226,265,308).  visible(mc1gt_1,308). happensAt(walking(mc1gt_1),308). orientation(mc1gt_3,172,308). coords(mc1gt_3,294,196,308).  visible(mc1gt_3,308). happensAt(walking(mc1gt_3),308).
orientation(mc1gt_1,14,309). coords(mc1gt_1,226,267,309).  visible(mc1gt_1,309). happensAt(walking(mc1gt_1),309). orientation(mc1gt_3,3,309). coords(mc1gt_3,297,199,309).  visible(mc1gt_3,309). happensAt(walking(mc1gt_3),309).
orientation(mc1gt_1,20,310). coords(mc1gt_1,227,267,310).  visible(mc1gt_1,310). happensAt(walking(mc1gt_1),310). orientation(mc1gt_3,1,310). coords(mc1gt_3,298,201,310).  visible(mc1gt_3,310). happensAt(walking(mc1gt_3),310).
orientation(mc1gt_1,19,311). coords(mc1gt_1,227,267,311).  visible(mc1gt_1,311). happensAt(walking(mc1gt_1),311). orientation(mc1gt_3,6,311). coords(mc1gt_3,299,203,311).  visible(mc1gt_3,311). happensAt(walking(mc1gt_3),311).
orientation(mc1gt_1,32,312). coords(mc1gt_1,227,267,312).  visible(mc1gt_1,312). happensAt(walking(mc1gt_1),312). orientation(mc1gt_3,178,312). coords(mc1gt_3,300,204,312).  visible(mc1gt_3,312). happensAt(walking(mc1gt_3),312).
orientation(mc1gt_1,39,313). coords(mc1gt_1,228,268,313).  visible(mc1gt_1,313). happensAt(walking(mc1gt_1),313). orientation(mc1gt_3,4,313). coords(mc1gt_3,301,206,313).  visible(mc1gt_3,313). happensAt(walking(mc1gt_3),313).
orientation(mc1gt_1,43,314). coords(mc1gt_1,228,269,314).  visible(mc1gt_1,314). happensAt(walking(mc1gt_1),314). orientation(mc1gt_3,25,314). coords(mc1gt_3,301,208,314).  visible(mc1gt_3,314). happensAt(walking(mc1gt_3),314).
orientation(mc1gt_1,49,315). coords(mc1gt_1,228,270,315).  visible(mc1gt_1,315). happensAt(walking(mc1gt_1),315). orientation(mc1gt_3,27,315). coords(mc1gt_3,302,210,315).  visible(mc1gt_3,315). happensAt(walking(mc1gt_3),315).
orientation(mc1gt_1,49,316). coords(mc1gt_1,228,272,316).  visible(mc1gt_1,316). happensAt(walking(mc1gt_1),316). orientation(mc1gt_3,6,316). coords(mc1gt_3,305,215,316).  visible(mc1gt_3,316). happensAt(walking(mc1gt_3),316).
orientation(mc1gt_1,32,317). coords(mc1gt_1,228,275,317).  visible(mc1gt_1,317). happensAt(walking(mc1gt_1),317). orientation(mc1gt_3,3,317). coords(mc1gt_3,306,219,317).  visible(mc1gt_3,317). happensAt(walking(mc1gt_3),317).
orientation(mc1gt_1,28,318). coords(mc1gt_1,229,277,318).  visible(mc1gt_1,318). happensAt(walking(mc1gt_1),318). orientation(mc1gt_3,2,318). coords(mc1gt_3,307,223,318).  visible(mc1gt_3,318). happensAt(walking(mc1gt_3),318).
orientation(mc1gt_1,30,319). coords(mc1gt_1,232,279,319).  visible(mc1gt_1,319). happensAt(walking(mc1gt_1),319). orientation(mc1gt_3,2,319). coords(mc1gt_3,308,225,319).  visible(mc1gt_3,319). happensAt(walking(mc1gt_3),319).
orientation(mc1gt_1,17,320). coords(mc1gt_1,235,280,320).  visible(mc1gt_1,320). happensAt(walking(mc1gt_1),320). orientation(mc1gt_3,3,320). coords(mc1gt_3,309,228,320).  visible(mc1gt_3,320). happensAt(walking(mc1gt_3),320).
orientation(mc1gt_1,48,321). coords(mc1gt_1,236,280,321).  visible(mc1gt_1,321). happensAt(walking(mc1gt_1),321). orientation(mc1gt_3,14,321). coords(mc1gt_3,310,231,321).  visible(mc1gt_3,321). happensAt(walking(mc1gt_3),321).
orientation(mc1gt_1,65,322). coords(mc1gt_1,243,280,322).  visible(mc1gt_1,322). happensAt(walking(mc1gt_1),322). orientation(mc1gt_3,26,322). coords(mc1gt_3,310,234,322).  visible(mc1gt_3,322). happensAt(walking(mc1gt_3),322).
orientation(mc1gt_1,65,323). coords(mc1gt_1,243,280,323).  visible(mc1gt_1,323). happensAt(walking(mc1gt_1),323). orientation(mc1gt_3,34,323). coords(mc1gt_3,310,235,323).  visible(mc1gt_3,323). happensAt(walking(mc1gt_3),323).
orientation(mc1gt_1,65,324). coords(mc1gt_1,243,280,324).  visible(mc1gt_1,324). happensAt(walking(mc1gt_1),324). orientation(mc1gt_3,42,324). coords(mc1gt_3,311,236,324).  visible(mc1gt_3,324). happensAt(walking(mc1gt_3),324).
#show holdsAt/2.