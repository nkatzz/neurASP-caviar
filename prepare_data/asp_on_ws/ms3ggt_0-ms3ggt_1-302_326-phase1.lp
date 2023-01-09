
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

	orientation(ms3ggt_0,159,302). coords(ms3ggt_0,81,144,302).  visible(ms3ggt_0,302). happensAt(walking(ms3ggt_0),302). orientation(ms3ggt_1,158,302). coords(ms3ggt_1,78,164,302).  visible(ms3ggt_1,302). happensAt(walking(ms3ggt_1),302).
orientation(ms3ggt_0,159,303). coords(ms3ggt_0,82,144,303).  visible(ms3ggt_0,303). happensAt(walking(ms3ggt_0),303). orientation(ms3ggt_1,158,303). coords(ms3ggt_1,78,164,303).  visible(ms3ggt_1,303). happensAt(walking(ms3ggt_1),303).
orientation(ms3ggt_0,159,304). coords(ms3ggt_0,83,144,304).  visible(ms3ggt_0,304). happensAt(walking(ms3ggt_0),304). orientation(ms3ggt_1,158,304). coords(ms3ggt_1,78,164,304).  visible(ms3ggt_1,304). happensAt(walking(ms3ggt_1),304).
orientation(ms3ggt_0,159,305). coords(ms3ggt_0,84,144,305).  visible(ms3ggt_0,305). happensAt(walking(ms3ggt_0),305). orientation(ms3ggt_1,158,305). coords(ms3ggt_1,79,164,305).  visible(ms3ggt_1,305). happensAt(walking(ms3ggt_1),305).
orientation(ms3ggt_0,161,306). coords(ms3ggt_0,84,144,306).  visible(ms3ggt_0,306). happensAt(walking(ms3ggt_0),306). orientation(ms3ggt_1,158,306). coords(ms3ggt_1,79,164,306).  visible(ms3ggt_1,306). happensAt(walking(ms3ggt_1),306).
orientation(ms3ggt_0,161,307). coords(ms3ggt_0,85,144,307).  visible(ms3ggt_0,307). happensAt(walking(ms3ggt_0),307). orientation(ms3ggt_1,160,307). coords(ms3ggt_1,81,164,307).  visible(ms3ggt_1,307). happensAt(walking(ms3ggt_1),307).
orientation(ms3ggt_0,161,308). coords(ms3ggt_0,85,144,308).  visible(ms3ggt_0,308). happensAt(walking(ms3ggt_0),308). orientation(ms3ggt_1,160,308). coords(ms3ggt_1,82,164,308).  visible(ms3ggt_1,308). happensAt(walking(ms3ggt_1),308).
orientation(ms3ggt_0,161,309). coords(ms3ggt_0,86,144,309).  visible(ms3ggt_0,309). happensAt(walking(ms3ggt_0),309). orientation(ms3ggt_1,157,309). coords(ms3ggt_1,84,164,309).  visible(ms3ggt_1,309). happensAt(walking(ms3ggt_1),309).
orientation(ms3ggt_0,161,310). coords(ms3ggt_0,86,144,310).  visible(ms3ggt_0,310). happensAt(walking(ms3ggt_0),310). orientation(ms3ggt_1,157,310). coords(ms3ggt_1,84,164,310).  visible(ms3ggt_1,310). happensAt(walking(ms3ggt_1),310).
orientation(ms3ggt_0,158,311). coords(ms3ggt_0,86,144,311).  visible(ms3ggt_0,311). happensAt(walking(ms3ggt_0),311). orientation(ms3ggt_1,160,311). coords(ms3ggt_1,84,165,311).  visible(ms3ggt_1,311). happensAt(walking(ms3ggt_1),311).
orientation(ms3ggt_0,158,312). coords(ms3ggt_0,86,143,312).  visible(ms3ggt_0,312). happensAt(walking(ms3ggt_0),312). orientation(ms3ggt_1,160,312). coords(ms3ggt_1,84,165,312).  visible(ms3ggt_1,312). happensAt(walking(ms3ggt_1),312).
orientation(ms3ggt_0,158,313). coords(ms3ggt_0,86,143,313).  visible(ms3ggt_0,313). happensAt(walking(ms3ggt_0),313). orientation(ms3ggt_1,160,313). coords(ms3ggt_1,84,165,313).  visible(ms3ggt_1,313). happensAt(walking(ms3ggt_1),313).
orientation(ms3ggt_0,158,314). coords(ms3ggt_0,86,144,314).  visible(ms3ggt_0,314). happensAt(walking(ms3ggt_0),314). orientation(ms3ggt_1,160,314). coords(ms3ggt_1,84,166,314).  visible(ms3ggt_1,314). happensAt(walking(ms3ggt_1),314).
orientation(ms3ggt_0,158,315). coords(ms3ggt_0,87,144,315).  visible(ms3ggt_0,315). happensAt(walking(ms3ggt_0),315). orientation(ms3ggt_1,160,315). coords(ms3ggt_1,85,166,315).  visible(ms3ggt_1,315). happensAt(walking(ms3ggt_1),315).
orientation(ms3ggt_0,153,316). coords(ms3ggt_0,90,144,316).  visible(ms3ggt_0,316). happensAt(walking(ms3ggt_0),316). orientation(ms3ggt_1,160,316). coords(ms3ggt_1,85,166,316).  visible(ms3ggt_1,316). happensAt(walking(ms3ggt_1),316).
orientation(ms3ggt_0,153,317). coords(ms3ggt_0,91,145,317).  visible(ms3ggt_0,317). happensAt(walking(ms3ggt_0),317). orientation(ms3ggt_1,160,317). coords(ms3ggt_1,86,166,317).  visible(ms3ggt_1,317). happensAt(walking(ms3ggt_1),317).
orientation(ms3ggt_0,153,318). coords(ms3ggt_0,92,145,318).  visible(ms3ggt_0,318). happensAt(walking(ms3ggt_0),318). orientation(ms3ggt_1,160,318). coords(ms3ggt_1,88,166,318).  visible(ms3ggt_1,318). happensAt(walking(ms3ggt_1),318).
orientation(ms3ggt_0,153,319). coords(ms3ggt_0,93,145,319).  visible(ms3ggt_0,319). happensAt(walking(ms3ggt_0),319). orientation(ms3ggt_1,162,319). coords(ms3ggt_1,90,166,319).  visible(ms3ggt_1,319). happensAt(walking(ms3ggt_1),319).
orientation(ms3ggt_0,159,320). coords(ms3ggt_0,94,145,320).  visible(ms3ggt_0,320). happensAt(walking(ms3ggt_0),320). orientation(ms3ggt_1,162,320). coords(ms3ggt_1,90,166,320).  visible(ms3ggt_1,320). happensAt(walking(ms3ggt_1),320).
orientation(ms3ggt_0,154,321). coords(ms3ggt_0,94,145,321).  visible(ms3ggt_0,321). happensAt(walking(ms3ggt_0),321). orientation(ms3ggt_1,162,321). coords(ms3ggt_1,91,166,321).  visible(ms3ggt_1,321). happensAt(walking(ms3ggt_1),321).
orientation(ms3ggt_0,154,322). coords(ms3ggt_0,95,145,322).  visible(ms3ggt_0,322). happensAt(walking(ms3ggt_0),322). orientation(ms3ggt_1,162,322). coords(ms3ggt_1,92,166,322).  visible(ms3ggt_1,322). happensAt(walking(ms3ggt_1),322).
orientation(ms3ggt_0,154,323). coords(ms3ggt_0,95,145,323).  visible(ms3ggt_0,323). happensAt(walking(ms3ggt_0),323). orientation(ms3ggt_1,162,323). coords(ms3ggt_1,93,166,323).  visible(ms3ggt_1,323). happensAt(walking(ms3ggt_1),323).
orientation(ms3ggt_0,154,324). coords(ms3ggt_0,95,145,324).  visible(ms3ggt_0,324). happensAt(walking(ms3ggt_0),324). orientation(ms3ggt_1,162,324). coords(ms3ggt_1,93,166,324).  visible(ms3ggt_1,324). happensAt(walking(ms3ggt_1),324).
orientation(ms3ggt_0,155,325). coords(ms3ggt_0,95,145,325).  visible(ms3ggt_0,325). happensAt(walking(ms3ggt_0),325). orientation(ms3ggt_1,162,325). coords(ms3ggt_1,93,167,325).  visible(ms3ggt_1,325). happensAt(walking(ms3ggt_1),325).
orientation(ms3ggt_0,155,326). coords(ms3ggt_0,95,145,326).  visible(ms3ggt_0,326). happensAt(walking(ms3ggt_0),326). orientation(ms3ggt_1,162,326). coords(ms3ggt_1,93,167,326).  visible(ms3ggt_1,326). happensAt(walking(ms3ggt_1),326).
#show holdsAt/2.