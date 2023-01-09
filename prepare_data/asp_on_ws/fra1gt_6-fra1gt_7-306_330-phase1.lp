
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

	orientation(fra1gt_6,133,306). coords(fra1gt_6,175,76,306).  visible(fra1gt_6,306). happensAt(walking(fra1gt_6),306). orientation(fra1gt_7,142,306). coords(fra1gt_7,138,62,306).  visible(fra1gt_7,306). happensAt(walking(fra1gt_7),306).
orientation(fra1gt_6,133,307). coords(fra1gt_6,172,76,307).  visible(fra1gt_6,307). happensAt(walking(fra1gt_6),307). orientation(fra1gt_7,142,307). coords(fra1gt_7,138,62,307).  visible(fra1gt_7,307). happensAt(walking(fra1gt_7),307).
orientation(fra1gt_6,133,308). coords(fra1gt_6,171,77,308).  visible(fra1gt_6,308). happensAt(walking(fra1gt_6),308). orientation(fra1gt_7,142,308). coords(fra1gt_7,138,62,308).  visible(fra1gt_7,308). happensAt(walking(fra1gt_7),308).
orientation(fra1gt_6,133,309). coords(fra1gt_6,169,78,309).  visible(fra1gt_6,309). happensAt(walking(fra1gt_6),309). orientation(fra1gt_7,142,309). coords(fra1gt_7,138,63,309).  visible(fra1gt_7,309). happensAt(walking(fra1gt_7),309).
orientation(fra1gt_6,133,310). coords(fra1gt_6,168,78,310).  visible(fra1gt_6,310). happensAt(walking(fra1gt_6),310). orientation(fra1gt_7,142,310). coords(fra1gt_7,138,63,310).  visible(fra1gt_7,310). happensAt(walking(fra1gt_7),310).
orientation(fra1gt_6,133,311). coords(fra1gt_6,167,78,311).  visible(fra1gt_6,311). happensAt(walking(fra1gt_6),311). orientation(fra1gt_7,142,311). coords(fra1gt_7,139,63,311).  visible(fra1gt_7,311). happensAt(walking(fra1gt_7),311).
orientation(fra1gt_6,133,312). coords(fra1gt_6,167,77,312).  visible(fra1gt_6,312). happensAt(walking(fra1gt_6),312). orientation(fra1gt_7,142,312). coords(fra1gt_7,139,64,312).  visible(fra1gt_7,312). happensAt(walking(fra1gt_7),312).
orientation(fra1gt_6,133,313). coords(fra1gt_6,166,78,313).  visible(fra1gt_6,313). happensAt(walking(fra1gt_6),313). orientation(fra1gt_7,142,313). coords(fra1gt_7,140,64,313).  visible(fra1gt_7,313). happensAt(walking(fra1gt_7),313).
orientation(fra1gt_6,133,314). coords(fra1gt_6,166,78,314).  visible(fra1gt_6,314). happensAt(walking(fra1gt_6),314). orientation(fra1gt_7,142,314). coords(fra1gt_7,140,64,314).  visible(fra1gt_7,314). happensAt(walking(fra1gt_7),314).
orientation(fra1gt_6,133,315). coords(fra1gt_6,165,79,315).  visible(fra1gt_6,315). happensAt(walking(fra1gt_6),315). orientation(fra1gt_7,142,315). coords(fra1gt_7,140,64,315).  visible(fra1gt_7,315). happensAt(walking(fra1gt_7),315).
orientation(fra1gt_6,137,316). coords(fra1gt_6,165,79,316).  visible(fra1gt_6,316). happensAt(walking(fra1gt_6),316). orientation(fra1gt_7,142,316). coords(fra1gt_7,140,64,316).  visible(fra1gt_7,316). happensAt(walking(fra1gt_7),316).
orientation(fra1gt_6,137,317). coords(fra1gt_6,164,79,317).  visible(fra1gt_6,317). happensAt(walking(fra1gt_6),317). orientation(fra1gt_7,142,317). coords(fra1gt_7,141,65,317).  visible(fra1gt_7,317). happensAt(walking(fra1gt_7),317).
orientation(fra1gt_6,137,318). coords(fra1gt_6,162,78,318).  visible(fra1gt_6,318). happensAt(walking(fra1gt_6),318). orientation(fra1gt_7,142,318). coords(fra1gt_7,141,65,318).  visible(fra1gt_7,318). happensAt(walking(fra1gt_7),318).
orientation(fra1gt_6,137,319). coords(fra1gt_6,162,78,319).  visible(fra1gt_6,319). happensAt(walking(fra1gt_6),319). orientation(fra1gt_7,142,319). coords(fra1gt_7,142,66,319).  visible(fra1gt_7,319). happensAt(walking(fra1gt_7),319).
orientation(fra1gt_6,137,320). coords(fra1gt_6,162,78,320).  visible(fra1gt_6,320). happensAt(walking(fra1gt_6),320). orientation(fra1gt_7,142,320). coords(fra1gt_7,142,66,320).  visible(fra1gt_7,320). happensAt(walking(fra1gt_7),320).
orientation(fra1gt_6,137,321). coords(fra1gt_6,159,77,321).  visible(fra1gt_6,321). happensAt(walking(fra1gt_6),321). orientation(fra1gt_7,142,321). coords(fra1gt_7,143,66,321).  visible(fra1gt_7,321). happensAt(walking(fra1gt_7),321).
orientation(fra1gt_6,137,322). coords(fra1gt_6,158,77,322).  visible(fra1gt_6,322). happensAt(walking(fra1gt_6),322). orientation(fra1gt_7,142,322). coords(fra1gt_7,142,64,322).  visible(fra1gt_7,322). happensAt(walking(fra1gt_7),322).
orientation(fra1gt_6,137,323). coords(fra1gt_6,156,77,323).  visible(fra1gt_6,323). happensAt(walking(fra1gt_6),323). orientation(fra1gt_7,142,323). coords(fra1gt_7,143,64,323).  visible(fra1gt_7,323). happensAt(walking(fra1gt_7),323).
orientation(fra1gt_6,137,324). coords(fra1gt_6,156,76,324).  visible(fra1gt_6,324). happensAt(walking(fra1gt_6),324). orientation(fra1gt_7,142,324). coords(fra1gt_7,153,65,324).  visible(fra1gt_7,324). happensAt(walking(fra1gt_7),324).
orientation(fra1gt_6,137,325). coords(fra1gt_6,155,76,325).  visible(fra1gt_6,325). happensAt(walking(fra1gt_6),325). orientation(fra1gt_7,142,325). coords(fra1gt_7,153,65,325).  visible(fra1gt_7,325). happensAt(walking(fra1gt_7),325).
orientation(fra1gt_6,137,326). coords(fra1gt_6,156,76,326).  visible(fra1gt_6,326). happensAt(walking(fra1gt_6),326). orientation(fra1gt_7,142,326). coords(fra1gt_7,151,65,326).  visible(fra1gt_7,326). happensAt(walking(fra1gt_7),326).
orientation(fra1gt_6,137,327). coords(fra1gt_6,156,76,327).  visible(fra1gt_6,327). happensAt(walking(fra1gt_6),327). orientation(fra1gt_7,142,327). coords(fra1gt_7,151,64,327).  visible(fra1gt_7,327). happensAt(walking(fra1gt_7),327).
orientation(fra1gt_6,137,328). coords(fra1gt_6,155,76,328).  visible(fra1gt_6,328). happensAt(walking(fra1gt_6),328). orientation(fra1gt_7,142,328). coords(fra1gt_7,151,64,328).  visible(fra1gt_7,328). happensAt(walking(fra1gt_7),328).
orientation(fra1gt_6,137,329). coords(fra1gt_6,155,76,329).  visible(fra1gt_6,329). happensAt(walking(fra1gt_6),329). orientation(fra1gt_7,142,329). coords(fra1gt_7,154,65,329).  visible(fra1gt_7,329). happensAt(walking(fra1gt_7),329).
orientation(fra1gt_6,134,330). coords(fra1gt_6,155,76,330).  visible(fra1gt_6,330). happensAt(walking(fra1gt_6),330). orientation(fra1gt_7,142,330). coords(fra1gt_7,157,65,330).  visible(fra1gt_7,330). happensAt(walking(fra1gt_7),330).
#show holdsAt/2.