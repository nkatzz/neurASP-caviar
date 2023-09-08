
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

	orientation(mws1gt_3,158,303). coords(mws1gt_3,72,135,303).  visible(mws1gt_3,303). happensAt(walking(mws1gt_3),303). orientation(mws1gt_5,168,303). coords(mws1gt_5,36,179,303).  visible(mws1gt_5,303). happensAt(walking(mws1gt_5),303).
orientation(mws1gt_3,158,304). coords(mws1gt_3,72,135,304).  visible(mws1gt_3,304). happensAt(walking(mws1gt_3),304). orientation(mws1gt_5,168,304). coords(mws1gt_5,36,179,304).  visible(mws1gt_5,304). happensAt(walking(mws1gt_5),304).
orientation(mws1gt_3,158,305). coords(mws1gt_3,72,136,305).  visible(mws1gt_3,305). happensAt(walking(mws1gt_3),305). orientation(mws1gt_5,168,305). coords(mws1gt_5,36,180,305).  visible(mws1gt_5,305). happensAt(walking(mws1gt_5),305).
orientation(mws1gt_3,158,306). coords(mws1gt_3,71,136,306).  visible(mws1gt_3,306). happensAt(walking(mws1gt_3),306). orientation(mws1gt_5,168,306). coords(mws1gt_5,36,181,306).  visible(mws1gt_5,306). happensAt(walking(mws1gt_5),306).
orientation(mws1gt_3,158,307). coords(mws1gt_3,71,136,307).  visible(mws1gt_3,307). happensAt(walking(mws1gt_3),307). orientation(mws1gt_5,168,307). coords(mws1gt_5,36,182,307).  visible(mws1gt_5,307). happensAt(walking(mws1gt_5),307).
orientation(mws1gt_3,158,308). coords(mws1gt_3,70,137,308).  visible(mws1gt_3,308). happensAt(walking(mws1gt_3),308). orientation(mws1gt_5,168,308). coords(mws1gt_5,35,183,308).  visible(mws1gt_5,308). happensAt(walking(mws1gt_5),308).
orientation(mws1gt_3,158,309). coords(mws1gt_3,70,137,309).  visible(mws1gt_3,309). happensAt(walking(mws1gt_3),309). orientation(mws1gt_5,168,309). coords(mws1gt_5,35,185,309).  visible(mws1gt_5,309). happensAt(walking(mws1gt_5),309).
orientation(mws1gt_3,158,310). coords(mws1gt_3,69,137,310).  visible(mws1gt_3,310). happensAt(walking(mws1gt_3),310). orientation(mws1gt_5,168,310). coords(mws1gt_5,35,186,310).  visible(mws1gt_5,310). happensAt(walking(mws1gt_5),310).
orientation(mws1gt_3,158,311). coords(mws1gt_3,68,137,311).  visible(mws1gt_3,311). happensAt(walking(mws1gt_3),311). orientation(mws1gt_5,168,311). coords(mws1gt_5,35,188,311).  visible(mws1gt_5,311). happensAt(walking(mws1gt_5),311).
orientation(mws1gt_3,158,312). coords(mws1gt_3,67,138,312).  visible(mws1gt_3,312). happensAt(walking(mws1gt_3),312). orientation(mws1gt_5,168,312). coords(mws1gt_5,35,189,312).  visible(mws1gt_5,312). happensAt(walking(mws1gt_5),312).
orientation(mws1gt_3,158,313). coords(mws1gt_3,66,138,313).  visible(mws1gt_3,313). happensAt(walking(mws1gt_3),313). orientation(mws1gt_5,168,313). coords(mws1gt_5,35,190,313).  visible(mws1gt_5,313). happensAt(walking(mws1gt_5),313).
orientation(mws1gt_3,158,314). coords(mws1gt_3,66,138,314).  visible(mws1gt_3,314). happensAt(walking(mws1gt_3),314). orientation(mws1gt_5,168,314). coords(mws1gt_5,35,190,314).  visible(mws1gt_5,314). happensAt(walking(mws1gt_5),314).
orientation(mws1gt_3,158,315). coords(mws1gt_3,66,139,315).  visible(mws1gt_3,315). happensAt(walking(mws1gt_3),315). orientation(mws1gt_5,168,315). coords(mws1gt_5,35,190,315).  visible(mws1gt_5,315). happensAt(walking(mws1gt_5),315).
orientation(mws1gt_3,158,316). coords(mws1gt_3,66,139,316).  visible(mws1gt_3,316). happensAt(walking(mws1gt_3),316). orientation(mws1gt_5,168,316). coords(mws1gt_5,35,191,316).  visible(mws1gt_5,316). happensAt(walking(mws1gt_5),316).
orientation(mws1gt_3,158,317). coords(mws1gt_3,66,139,317).  visible(mws1gt_3,317). happensAt(walking(mws1gt_3),317). orientation(mws1gt_5,168,317). coords(mws1gt_5,35,191,317).  visible(mws1gt_5,317). happensAt(walking(mws1gt_5),317).
orientation(mws1gt_3,158,318). coords(mws1gt_3,65,139,318).  visible(mws1gt_3,318). happensAt(walking(mws1gt_3),318). orientation(mws1gt_5,168,318). coords(mws1gt_5,35,193,318).  visible(mws1gt_5,318). happensAt(walking(mws1gt_5),318).
orientation(mws1gt_3,158,319). coords(mws1gt_3,65,140,319).  visible(mws1gt_3,319). happensAt(walking(mws1gt_3),319). orientation(mws1gt_5,168,319). coords(mws1gt_5,35,195,319).  visible(mws1gt_5,319). happensAt(walking(mws1gt_5),319).
orientation(mws1gt_3,158,320). coords(mws1gt_3,64,140,320).  visible(mws1gt_3,320). happensAt(walking(mws1gt_3),320). orientation(mws1gt_5,168,320). coords(mws1gt_5,35,197,320).  visible(mws1gt_5,320). happensAt(walking(mws1gt_5),320).
orientation(mws1gt_3,158,321). coords(mws1gt_3,64,140,321).  visible(mws1gt_3,321). happensAt(walking(mws1gt_3),321). orientation(mws1gt_5,168,321). coords(mws1gt_5,35,198,321).  visible(mws1gt_5,321). happensAt(walking(mws1gt_5),321).
orientation(mws1gt_3,165,322). coords(mws1gt_3,64,140,322).  visible(mws1gt_3,322). happensAt(walking(mws1gt_3),322). orientation(mws1gt_5,168,322). coords(mws1gt_5,35,200,322).  visible(mws1gt_5,322). happensAt(walking(mws1gt_5),322).
orientation(mws1gt_3,165,323). coords(mws1gt_3,63,141,323).  visible(mws1gt_3,323). happensAt(walking(mws1gt_3),323). orientation(mws1gt_5,168,323). coords(mws1gt_5,33,206,323).  visible(mws1gt_5,323). happensAt(walking(mws1gt_5),323).
orientation(mws1gt_3,165,324). coords(mws1gt_3,63,141,324).  visible(mws1gt_3,324). happensAt(walking(mws1gt_3),324). orientation(mws1gt_5,168,324). coords(mws1gt_5,33,206,324).  visible(mws1gt_5,324). happensAt(walking(mws1gt_5),324).
orientation(mws1gt_3,165,325). coords(mws1gt_3,63,141,325).  visible(mws1gt_3,325). happensAt(walking(mws1gt_3),325). orientation(mws1gt_5,168,325). coords(mws1gt_5,33,206,325).  visible(mws1gt_5,325). happensAt(walking(mws1gt_5),325).
orientation(mws1gt_3,165,326). coords(mws1gt_3,63,142,326).  visible(mws1gt_3,326). happensAt(walking(mws1gt_3),326). orientation(mws1gt_5,168,326). coords(mws1gt_5,32,207,326).  visible(mws1gt_5,326). happensAt(walking(mws1gt_5),326).
orientation(mws1gt_3,165,327). coords(mws1gt_3,62,143,327).  visible(mws1gt_3,327). happensAt(walking(mws1gt_3),327). orientation(mws1gt_5,168,327). coords(mws1gt_5,31,208,327).  happensAt(disappear(mws1gt_5),327). happensAt(walking(mws1gt_5),327).
#show holdsAt/2.