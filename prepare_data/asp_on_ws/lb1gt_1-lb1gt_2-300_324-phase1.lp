
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

	orientation(lb1gt_1,151,300). coords(lb1gt_1,160,164,300).  visible(lb1gt_1,300). happensAt(walking(lb1gt_1),300). orientation(lb1gt_2,146,300). coords(lb1gt_2,194,152,300).  visible(lb1gt_2,300). happensAt(active(lb1gt_2),300).
orientation(lb1gt_1,151,301). coords(lb1gt_1,160,164,301).  visible(lb1gt_1,301). happensAt(walking(lb1gt_1),301). orientation(lb1gt_2,146,301). coords(lb1gt_2,194,152,301).  visible(lb1gt_2,301). happensAt(active(lb1gt_2),301).
orientation(lb1gt_1,151,302). coords(lb1gt_1,161,165,302).  visible(lb1gt_1,302). happensAt(walking(lb1gt_1),302). orientation(lb1gt_2,146,302). coords(lb1gt_2,194,152,302).  visible(lb1gt_2,302). happensAt(active(lb1gt_2),302).
orientation(lb1gt_1,151,303). coords(lb1gt_1,161,165,303).  visible(lb1gt_1,303). happensAt(walking(lb1gt_1),303). orientation(lb1gt_2,146,303). coords(lb1gt_2,194,152,303).  visible(lb1gt_2,303). happensAt(active(lb1gt_2),303).
orientation(lb1gt_1,151,304). coords(lb1gt_1,161,166,304).  visible(lb1gt_1,304). happensAt(walking(lb1gt_1),304). orientation(lb1gt_2,146,304). coords(lb1gt_2,194,152,304).  visible(lb1gt_2,304). happensAt(active(lb1gt_2),304).
orientation(lb1gt_1,151,305). coords(lb1gt_1,161,166,305).  visible(lb1gt_1,305). happensAt(walking(lb1gt_1),305). orientation(lb1gt_2,146,305). coords(lb1gt_2,194,152,305).  visible(lb1gt_2,305). happensAt(active(lb1gt_2),305).
orientation(lb1gt_1,151,306). coords(lb1gt_1,161,167,306).  visible(lb1gt_1,306). happensAt(walking(lb1gt_1),306). orientation(lb1gt_2,146,306). coords(lb1gt_2,194,152,306).  visible(lb1gt_2,306). happensAt(active(lb1gt_2),306).
orientation(lb1gt_1,151,307). coords(lb1gt_1,161,167,307).  visible(lb1gt_1,307). happensAt(walking(lb1gt_1),307). orientation(lb1gt_2,146,307). coords(lb1gt_2,194,152,307).  visible(lb1gt_2,307). happensAt(active(lb1gt_2),307).
orientation(lb1gt_1,151,308). coords(lb1gt_1,161,168,308).  visible(lb1gt_1,308). happensAt(walking(lb1gt_1),308). orientation(lb1gt_2,146,308). coords(lb1gt_2,194,152,308).  visible(lb1gt_2,308). happensAt(active(lb1gt_2),308).
orientation(lb1gt_1,151,309). coords(lb1gt_1,161,168,309).  visible(lb1gt_1,309). happensAt(walking(lb1gt_1),309). orientation(lb1gt_2,146,309). coords(lb1gt_2,194,151,309).  visible(lb1gt_2,309). happensAt(active(lb1gt_2),309).
orientation(lb1gt_1,151,310). coords(lb1gt_1,161,168,310).  visible(lb1gt_1,310). happensAt(walking(lb1gt_1),310). orientation(lb1gt_2,146,310). coords(lb1gt_2,193,151,310).  visible(lb1gt_2,310). happensAt(active(lb1gt_2),310).
orientation(lb1gt_1,151,311). coords(lb1gt_1,162,169,311).  visible(lb1gt_1,311). happensAt(walking(lb1gt_1),311). orientation(lb1gt_2,146,311). coords(lb1gt_2,193,151,311).  visible(lb1gt_2,311). happensAt(active(lb1gt_2),311).
orientation(lb1gt_1,151,312). coords(lb1gt_1,162,169,312).  visible(lb1gt_1,312). happensAt(walking(lb1gt_1),312). orientation(lb1gt_2,146,312). coords(lb1gt_2,193,151,312).  visible(lb1gt_2,312). happensAt(active(lb1gt_2),312).
orientation(lb1gt_1,151,313). coords(lb1gt_1,162,170,313).  visible(lb1gt_1,313). happensAt(walking(lb1gt_1),313). orientation(lb1gt_2,146,313). coords(lb1gt_2,193,150,313).  visible(lb1gt_2,313). happensAt(active(lb1gt_2),313).
orientation(lb1gt_1,151,314). coords(lb1gt_1,162,170,314).  visible(lb1gt_1,314). happensAt(walking(lb1gt_1),314). orientation(lb1gt_2,146,314). coords(lb1gt_2,192,150,314).  visible(lb1gt_2,314). happensAt(active(lb1gt_2),314).
orientation(lb1gt_1,151,315). coords(lb1gt_1,162,170,315).  visible(lb1gt_1,315). happensAt(walking(lb1gt_1),315). orientation(lb1gt_2,146,315). coords(lb1gt_2,191,150,315).  visible(lb1gt_2,315). happensAt(active(lb1gt_2),315).
orientation(lb1gt_1,151,316). coords(lb1gt_1,163,170,316).  visible(lb1gt_1,316). happensAt(walking(lb1gt_1),316). orientation(lb1gt_2,146,316). coords(lb1gt_2,191,150,316).  visible(lb1gt_2,316). happensAt(active(lb1gt_2),316).
orientation(lb1gt_1,151,317). coords(lb1gt_1,163,170,317).  visible(lb1gt_1,317). happensAt(walking(lb1gt_1),317). orientation(lb1gt_2,146,317). coords(lb1gt_2,191,150,317).  visible(lb1gt_2,317). happensAt(active(lb1gt_2),317).
orientation(lb1gt_1,151,318). coords(lb1gt_1,164,170,318).  visible(lb1gt_1,318). happensAt(walking(lb1gt_1),318). orientation(lb1gt_2,146,318). coords(lb1gt_2,190,150,318).  visible(lb1gt_2,318). happensAt(active(lb1gt_2),318).
orientation(lb1gt_1,151,319). coords(lb1gt_1,164,170,319).  visible(lb1gt_1,319). happensAt(walking(lb1gt_1),319). orientation(lb1gt_2,146,319). coords(lb1gt_2,190,150,319).  visible(lb1gt_2,319). happensAt(active(lb1gt_2),319).
orientation(lb1gt_1,151,320). coords(lb1gt_1,165,170,320).  visible(lb1gt_1,320). happensAt(walking(lb1gt_1),320). orientation(lb1gt_2,146,320). coords(lb1gt_2,190,150,320).  visible(lb1gt_2,320). happensAt(active(lb1gt_2),320).
orientation(lb1gt_1,151,321). coords(lb1gt_1,165,170,321).  visible(lb1gt_1,321). happensAt(walking(lb1gt_1),321). orientation(lb1gt_2,146,321). coords(lb1gt_2,190,150,321).  visible(lb1gt_2,321). happensAt(active(lb1gt_2),321).
orientation(lb1gt_1,151,322). coords(lb1gt_1,165,170,322).  visible(lb1gt_1,322). happensAt(walking(lb1gt_1),322). orientation(lb1gt_2,146,322). coords(lb1gt_2,190,150,322).  visible(lb1gt_2,322). happensAt(active(lb1gt_2),322).
orientation(lb1gt_1,151,323). coords(lb1gt_1,165,170,323).  visible(lb1gt_1,323). happensAt(walking(lb1gt_1),323). orientation(lb1gt_2,146,323). coords(lb1gt_2,190,150,323).  visible(lb1gt_2,323). happensAt(active(lb1gt_2),323).
orientation(lb1gt_1,151,324). coords(lb1gt_1,165,170,324).  visible(lb1gt_1,324). happensAt(active(lb1gt_1),324). orientation(lb1gt_2,146,324). coords(lb1gt_2,190,150,324).  visible(lb1gt_2,324). happensAt(active(lb1gt_2),324).
#show holdsAt/2.