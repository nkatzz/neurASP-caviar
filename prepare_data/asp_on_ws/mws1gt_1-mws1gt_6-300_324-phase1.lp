
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

	orientation(mws1gt_1,155,300). coords(mws1gt_1,85,104,300).  visible(mws1gt_1,300). happensAt(walking(mws1gt_1),300). orientation(mws1gt_6,146,300). coords(mws1gt_6,36,155,300).  visible(mws1gt_6,300). happensAt(walking(mws1gt_6),300).
orientation(mws1gt_1,155,301). coords(mws1gt_1,84,103,301).  visible(mws1gt_1,301). happensAt(walking(mws1gt_1),301). orientation(mws1gt_6,146,301). coords(mws1gt_6,36,155,301).  visible(mws1gt_6,301). happensAt(walking(mws1gt_6),301).
orientation(mws1gt_1,155,302). coords(mws1gt_1,83,103,302).  visible(mws1gt_1,302). happensAt(walking(mws1gt_1),302). orientation(mws1gt_6,146,302). coords(mws1gt_6,36,155,302).  visible(mws1gt_6,302). happensAt(walking(mws1gt_6),302).
orientation(mws1gt_1,155,303). coords(mws1gt_1,82,103,303).  visible(mws1gt_1,303). happensAt(walking(mws1gt_1),303). orientation(mws1gt_6,146,303). coords(mws1gt_6,36,156,303).  visible(mws1gt_6,303). happensAt(walking(mws1gt_6),303).
orientation(mws1gt_1,155,304). coords(mws1gt_1,82,103,304).  visible(mws1gt_1,304). happensAt(walking(mws1gt_1),304). orientation(mws1gt_6,146,304). coords(mws1gt_6,35,157,304).  visible(mws1gt_6,304). happensAt(walking(mws1gt_6),304).
orientation(mws1gt_1,155,305). coords(mws1gt_1,81,103,305).  visible(mws1gt_1,305). happensAt(walking(mws1gt_1),305). orientation(mws1gt_6,146,305). coords(mws1gt_6,35,159,305).  visible(mws1gt_6,305). happensAt(walking(mws1gt_6),305).
orientation(mws1gt_1,155,306). coords(mws1gt_1,81,103,306).  visible(mws1gt_1,306). happensAt(walking(mws1gt_1),306). orientation(mws1gt_6,146,306). coords(mws1gt_6,35,160,306).  visible(mws1gt_6,306). happensAt(walking(mws1gt_6),306).
orientation(mws1gt_1,155,307). coords(mws1gt_1,80,103,307).  visible(mws1gt_1,307). happensAt(walking(mws1gt_1),307). orientation(mws1gt_6,146,307). coords(mws1gt_6,35,162,307).  visible(mws1gt_6,307). happensAt(walking(mws1gt_6),307).
orientation(mws1gt_1,155,308). coords(mws1gt_1,80,103,308).  visible(mws1gt_1,308). happensAt(walking(mws1gt_1),308). orientation(mws1gt_6,146,308). coords(mws1gt_6,35,163,308).  visible(mws1gt_6,308). happensAt(walking(mws1gt_6),308).
orientation(mws1gt_1,155,309). coords(mws1gt_1,80,103,309).  visible(mws1gt_1,309). happensAt(walking(mws1gt_1),309). orientation(mws1gt_6,146,309). coords(mws1gt_6,35,164,309).  visible(mws1gt_6,309). happensAt(walking(mws1gt_6),309).
orientation(mws1gt_1,155,310). coords(mws1gt_1,80,103,310).  visible(mws1gt_1,310). happensAt(walking(mws1gt_1),310). orientation(mws1gt_6,146,310). coords(mws1gt_6,35,164,310).  visible(mws1gt_6,310). happensAt(walking(mws1gt_6),310).
orientation(mws1gt_1,155,311). coords(mws1gt_1,80,103,311).  visible(mws1gt_1,311). happensAt(walking(mws1gt_1),311). orientation(mws1gt_6,146,311). coords(mws1gt_6,35,165,311).  visible(mws1gt_6,311). happensAt(walking(mws1gt_6),311).
orientation(mws1gt_1,155,312). coords(mws1gt_1,80,102,312).  visible(mws1gt_1,312). happensAt(walking(mws1gt_1),312). orientation(mws1gt_6,146,312). coords(mws1gt_6,35,166,312).  visible(mws1gt_6,312). happensAt(walking(mws1gt_6),312).
orientation(mws1gt_1,155,313). coords(mws1gt_1,80,102,313).  visible(mws1gt_1,313). happensAt(walking(mws1gt_1),313). orientation(mws1gt_6,146,313). coords(mws1gt_6,35,166,313).  visible(mws1gt_6,313). happensAt(walking(mws1gt_6),313).
orientation(mws1gt_1,155,314). coords(mws1gt_1,80,102,314).  visible(mws1gt_1,314). happensAt(walking(mws1gt_1),314). orientation(mws1gt_6,146,314). coords(mws1gt_6,35,166,314).  visible(mws1gt_6,314). happensAt(walking(mws1gt_6),314).
orientation(mws1gt_1,155,315). coords(mws1gt_1,78,102,315).  visible(mws1gt_1,315). happensAt(walking(mws1gt_1),315). orientation(mws1gt_6,146,315). coords(mws1gt_6,35,167,315).  visible(mws1gt_6,315). happensAt(walking(mws1gt_6),315).
orientation(mws1gt_1,155,316). coords(mws1gt_1,78,101,316).  visible(mws1gt_1,316). happensAt(walking(mws1gt_1),316). orientation(mws1gt_6,146,316). coords(mws1gt_6,35,167,316).  visible(mws1gt_6,316). happensAt(walking(mws1gt_6),316).
orientation(mws1gt_1,155,317). coords(mws1gt_1,76,101,317).  visible(mws1gt_1,317). happensAt(walking(mws1gt_1),317). orientation(mws1gt_6,146,317). coords(mws1gt_6,35,167,317).  visible(mws1gt_6,317). happensAt(walking(mws1gt_6),317).
orientation(mws1gt_1,155,318). coords(mws1gt_1,76,100,318).  visible(mws1gt_1,318). happensAt(walking(mws1gt_1),318). orientation(mws1gt_6,146,318). coords(mws1gt_6,35,168,318).  visible(mws1gt_6,318). happensAt(walking(mws1gt_6),318).
orientation(mws1gt_1,155,319). coords(mws1gt_1,76,100,319).  visible(mws1gt_1,319). happensAt(walking(mws1gt_1),319). orientation(mws1gt_6,146,319). coords(mws1gt_6,35,169,319).  visible(mws1gt_6,319). happensAt(walking(mws1gt_6),319).
orientation(mws1gt_1,155,320). coords(mws1gt_1,76,100,320).  visible(mws1gt_1,320). happensAt(walking(mws1gt_1),320). orientation(mws1gt_6,146,320). coords(mws1gt_6,35,170,320).  visible(mws1gt_6,320). happensAt(walking(mws1gt_6),320).
orientation(mws1gt_1,155,321). coords(mws1gt_1,76,100,321).  visible(mws1gt_1,321). happensAt(walking(mws1gt_1),321). orientation(mws1gt_6,146,321). coords(mws1gt_6,34,172,321).  visible(mws1gt_6,321). happensAt(walking(mws1gt_6),321).
orientation(mws1gt_1,155,322). coords(mws1gt_1,76,100,322).  visible(mws1gt_1,322). happensAt(walking(mws1gt_1),322). orientation(mws1gt_6,146,322). coords(mws1gt_6,33,174,322).  visible(mws1gt_6,322). happensAt(walking(mws1gt_6),322).
orientation(mws1gt_1,155,323). coords(mws1gt_1,76,100,323).  visible(mws1gt_1,323). happensAt(walking(mws1gt_1),323). orientation(mws1gt_6,146,323). coords(mws1gt_6,33,176,323).  visible(mws1gt_6,323). happensAt(walking(mws1gt_6),323).
orientation(mws1gt_1,155,324). coords(mws1gt_1,75,100,324).  visible(mws1gt_1,324). happensAt(walking(mws1gt_1),324). orientation(mws1gt_6,146,324). coords(mws1gt_6,33,176,324).  visible(mws1gt_6,324). happensAt(walking(mws1gt_6),324).
#show holdsAt/2.