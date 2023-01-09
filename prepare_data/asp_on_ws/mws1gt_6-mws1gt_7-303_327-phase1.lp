
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

	orientation(mws1gt_6,146,303). coords(mws1gt_6,36,156,303).  visible(mws1gt_6,303). happensAt(walking(mws1gt_6),303). orientation(mws1gt_7,14,303). coords(mws1gt_7,28,211,303).  happensAt(appear(mws1gt_7),303). happensAt(walking(mws1gt_7),303).
orientation(mws1gt_6,146,304). coords(mws1gt_6,35,157,304).  visible(mws1gt_6,304). happensAt(walking(mws1gt_6),304). orientation(mws1gt_7,14,304). coords(mws1gt_7,28,210,304).  visible(mws1gt_7,304). happensAt(walking(mws1gt_7),304).
orientation(mws1gt_6,146,305). coords(mws1gt_6,35,159,305).  visible(mws1gt_6,305). happensAt(walking(mws1gt_6),305). orientation(mws1gt_7,14,305). coords(mws1gt_7,28,210,305).  visible(mws1gt_7,305). happensAt(walking(mws1gt_7),305).
orientation(mws1gt_6,146,306). coords(mws1gt_6,35,160,306).  visible(mws1gt_6,306). happensAt(walking(mws1gt_6),306). orientation(mws1gt_7,14,306). coords(mws1gt_7,28,210,306).  visible(mws1gt_7,306). happensAt(walking(mws1gt_7),306).
orientation(mws1gt_6,146,307). coords(mws1gt_6,35,162,307).  visible(mws1gt_6,307). happensAt(walking(mws1gt_6),307). orientation(mws1gt_7,14,307). coords(mws1gt_7,28,209,307).  visible(mws1gt_7,307). happensAt(walking(mws1gt_7),307).
orientation(mws1gt_6,146,308). coords(mws1gt_6,35,163,308).  visible(mws1gt_6,308). happensAt(walking(mws1gt_6),308). orientation(mws1gt_7,14,308). coords(mws1gt_7,28,209,308).  visible(mws1gt_7,308). happensAt(walking(mws1gt_7),308).
orientation(mws1gt_6,146,309). coords(mws1gt_6,35,164,309).  visible(mws1gt_6,309). happensAt(walking(mws1gt_6),309). orientation(mws1gt_7,14,309). coords(mws1gt_7,29,208,309).  visible(mws1gt_7,309). happensAt(walking(mws1gt_7),309).
orientation(mws1gt_6,146,310). coords(mws1gt_6,35,164,310).  visible(mws1gt_6,310). happensAt(walking(mws1gt_6),310). orientation(mws1gt_7,14,310). coords(mws1gt_7,29,208,310).  visible(mws1gt_7,310). happensAt(walking(mws1gt_7),310).
orientation(mws1gt_6,146,311). coords(mws1gt_6,35,165,311).  visible(mws1gt_6,311). happensAt(walking(mws1gt_6),311). orientation(mws1gt_7,14,311). coords(mws1gt_7,29,208,311).  visible(mws1gt_7,311). happensAt(walking(mws1gt_7),311).
orientation(mws1gt_6,146,312). coords(mws1gt_6,35,166,312).  visible(mws1gt_6,312). happensAt(walking(mws1gt_6),312). orientation(mws1gt_7,14,312). coords(mws1gt_7,31,207,312).  visible(mws1gt_7,312). happensAt(walking(mws1gt_7),312).
orientation(mws1gt_6,146,313). coords(mws1gt_6,35,166,313).  visible(mws1gt_6,313). happensAt(walking(mws1gt_6),313). orientation(mws1gt_7,14,313). coords(mws1gt_7,31,207,313).  visible(mws1gt_7,313). happensAt(walking(mws1gt_7),313).
orientation(mws1gt_6,146,314). coords(mws1gt_6,35,166,314).  visible(mws1gt_6,314). happensAt(walking(mws1gt_6),314). orientation(mws1gt_7,14,314). coords(mws1gt_7,31,206,314).  visible(mws1gt_7,314). happensAt(walking(mws1gt_7),314).
orientation(mws1gt_6,146,315). coords(mws1gt_6,35,167,315).  visible(mws1gt_6,315). happensAt(walking(mws1gt_6),315). orientation(mws1gt_7,14,315). coords(mws1gt_7,33,206,315).  visible(mws1gt_7,315). happensAt(walking(mws1gt_7),315).
orientation(mws1gt_6,146,316). coords(mws1gt_6,35,167,316).  visible(mws1gt_6,316). happensAt(walking(mws1gt_6),316). orientation(mws1gt_7,14,316). coords(mws1gt_7,33,206,316).  visible(mws1gt_7,316). happensAt(walking(mws1gt_7),316).
orientation(mws1gt_6,146,317). coords(mws1gt_6,35,167,317).  visible(mws1gt_6,317). happensAt(walking(mws1gt_6),317). orientation(mws1gt_7,14,317). coords(mws1gt_7,34,205,317).  visible(mws1gt_7,317). happensAt(walking(mws1gt_7),317).
orientation(mws1gt_6,146,318). coords(mws1gt_6,35,168,318).  visible(mws1gt_6,318). happensAt(walking(mws1gt_6),318). orientation(mws1gt_7,14,318). coords(mws1gt_7,36,205,318).  visible(mws1gt_7,318). happensAt(walking(mws1gt_7),318).
orientation(mws1gt_6,146,319). coords(mws1gt_6,35,169,319).  visible(mws1gt_6,319). happensAt(walking(mws1gt_6),319). orientation(mws1gt_7,14,319). coords(mws1gt_7,36,204,319).  visible(mws1gt_7,319). happensAt(walking(mws1gt_7),319).
orientation(mws1gt_6,146,320). coords(mws1gt_6,35,170,320).  visible(mws1gt_6,320). happensAt(walking(mws1gt_6),320). orientation(mws1gt_7,14,320). coords(mws1gt_7,36,204,320).  visible(mws1gt_7,320). happensAt(walking(mws1gt_7),320).
orientation(mws1gt_6,146,321). coords(mws1gt_6,34,172,321).  visible(mws1gt_6,321). happensAt(walking(mws1gt_6),321). orientation(mws1gt_7,14,321). coords(mws1gt_7,36,204,321).  visible(mws1gt_7,321). happensAt(walking(mws1gt_7),321).
orientation(mws1gt_6,146,322). coords(mws1gt_6,33,174,322).  visible(mws1gt_6,322). happensAt(walking(mws1gt_6),322). orientation(mws1gt_7,14,322). coords(mws1gt_7,37,202,322).  visible(mws1gt_7,322). happensAt(walking(mws1gt_7),322).
orientation(mws1gt_6,146,323). coords(mws1gt_6,33,176,323).  visible(mws1gt_6,323). happensAt(walking(mws1gt_6),323). orientation(mws1gt_7,14,323). coords(mws1gt_7,37,201,323).  visible(mws1gt_7,323). happensAt(walking(mws1gt_7),323).
orientation(mws1gt_6,146,324). coords(mws1gt_6,33,176,324).  visible(mws1gt_6,324). happensAt(walking(mws1gt_6),324). orientation(mws1gt_7,0,324). coords(mws1gt_7,38,200,324).  visible(mws1gt_7,324). happensAt(walking(mws1gt_7),324).
orientation(mws1gt_6,146,325). coords(mws1gt_6,33,176,325).  visible(mws1gt_6,325). happensAt(walking(mws1gt_6),325). orientation(mws1gt_7,0,325). coords(mws1gt_7,38,199,325).  visible(mws1gt_7,325). happensAt(walking(mws1gt_7),325).
orientation(mws1gt_6,146,326). coords(mws1gt_6,33,177,326).  visible(mws1gt_6,326). happensAt(walking(mws1gt_6),326). orientation(mws1gt_7,170,326). coords(mws1gt_7,38,199,326).  visible(mws1gt_7,326). happensAt(walking(mws1gt_7),326).
orientation(mws1gt_6,146,327). coords(mws1gt_6,33,178,327).  visible(mws1gt_6,327). happensAt(walking(mws1gt_6),327). orientation(mws1gt_7,162,327). coords(mws1gt_7,38,199,327).  visible(mws1gt_7,327). happensAt(walking(mws1gt_7),327).
#show holdsAt/2.