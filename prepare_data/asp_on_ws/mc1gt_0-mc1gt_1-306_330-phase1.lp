
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

	orientation(mc1gt_0,4,306). coords(mc1gt_0,202,234,306).  visible(mc1gt_0,306). happensAt(walking(mc1gt_0),306). orientation(mc1gt_1,12,306). coords(mc1gt_1,224,263,306).  visible(mc1gt_1,306). happensAt(walking(mc1gt_1),306).
orientation(mc1gt_0,4,307). coords(mc1gt_0,201,235,307).  visible(mc1gt_0,307). happensAt(walking(mc1gt_0),307). orientation(mc1gt_1,8,307). coords(mc1gt_1,225,264,307).  visible(mc1gt_1,307). happensAt(walking(mc1gt_1),307).
orientation(mc1gt_0,10,308). coords(mc1gt_0,201,237,308).  visible(mc1gt_0,308). happensAt(walking(mc1gt_0),308). orientation(mc1gt_1,15,308). coords(mc1gt_1,226,265,308).  visible(mc1gt_1,308). happensAt(walking(mc1gt_1),308).
orientation(mc1gt_0,10,309). coords(mc1gt_0,202,238,309).  visible(mc1gt_0,309). happensAt(walking(mc1gt_0),309). orientation(mc1gt_1,14,309). coords(mc1gt_1,226,267,309).  visible(mc1gt_1,309). happensAt(walking(mc1gt_1),309).
orientation(mc1gt_0,10,310). coords(mc1gt_0,202,242,310).  visible(mc1gt_0,310). happensAt(walking(mc1gt_0),310). orientation(mc1gt_1,20,310). coords(mc1gt_1,227,267,310).  visible(mc1gt_1,310). happensAt(walking(mc1gt_1),310).
orientation(mc1gt_0,10,311). coords(mc1gt_0,204,244,311).  visible(mc1gt_0,311). happensAt(walking(mc1gt_0),311). orientation(mc1gt_1,19,311). coords(mc1gt_1,227,267,311).  visible(mc1gt_1,311). happensAt(walking(mc1gt_1),311).
orientation(mc1gt_0,10,312). coords(mc1gt_0,206,247,312).  visible(mc1gt_0,312). happensAt(walking(mc1gt_0),312). orientation(mc1gt_1,32,312). coords(mc1gt_1,227,267,312).  visible(mc1gt_1,312). happensAt(walking(mc1gt_1),312).
orientation(mc1gt_0,10,313). coords(mc1gt_0,208,249,313).  visible(mc1gt_0,313). happensAt(walking(mc1gt_0),313). orientation(mc1gt_1,39,313). coords(mc1gt_1,228,268,313).  visible(mc1gt_1,313). happensAt(walking(mc1gt_1),313).
orientation(mc1gt_0,10,314). coords(mc1gt_0,209,251,314).  visible(mc1gt_0,314). happensAt(walking(mc1gt_0),314). orientation(mc1gt_1,43,314). coords(mc1gt_1,228,269,314).  visible(mc1gt_1,314). happensAt(walking(mc1gt_1),314).
orientation(mc1gt_0,10,315). coords(mc1gt_0,209,252,315).  visible(mc1gt_0,315). happensAt(walking(mc1gt_0),315). orientation(mc1gt_1,49,315). coords(mc1gt_1,228,270,315).  visible(mc1gt_1,315). happensAt(walking(mc1gt_1),315).
orientation(mc1gt_0,10,316). coords(mc1gt_0,210,253,316).  visible(mc1gt_0,316). happensAt(walking(mc1gt_0),316). orientation(mc1gt_1,49,316). coords(mc1gt_1,228,272,316).  visible(mc1gt_1,316). happensAt(walking(mc1gt_1),316).
orientation(mc1gt_0,10,317). coords(mc1gt_0,211,254,317).  visible(mc1gt_0,317). happensAt(walking(mc1gt_0),317). orientation(mc1gt_1,32,317). coords(mc1gt_1,228,275,317).  visible(mc1gt_1,317). happensAt(walking(mc1gt_1),317).
orientation(mc1gt_0,10,318). coords(mc1gt_0,211,255,318).  visible(mc1gt_0,318). happensAt(walking(mc1gt_0),318). orientation(mc1gt_1,28,318). coords(mc1gt_1,229,277,318).  visible(mc1gt_1,318). happensAt(walking(mc1gt_1),318).
orientation(mc1gt_0,10,319). coords(mc1gt_0,212,255,319).  visible(mc1gt_0,319). happensAt(walking(mc1gt_0),319). orientation(mc1gt_1,30,319). coords(mc1gt_1,232,279,319).  visible(mc1gt_1,319). happensAt(walking(mc1gt_1),319).
orientation(mc1gt_0,10,320). coords(mc1gt_0,213,255,320).  visible(mc1gt_0,320). happensAt(walking(mc1gt_0),320). orientation(mc1gt_1,17,320). coords(mc1gt_1,235,280,320).  visible(mc1gt_1,320). happensAt(walking(mc1gt_1),320).
orientation(mc1gt_0,10,321). coords(mc1gt_0,213,257,321).  visible(mc1gt_0,321). happensAt(walking(mc1gt_0),321). orientation(mc1gt_1,48,321). coords(mc1gt_1,236,280,321).  visible(mc1gt_1,321). happensAt(walking(mc1gt_1),321).
orientation(mc1gt_0,10,322). coords(mc1gt_0,213,258,322).  visible(mc1gt_0,322). happensAt(walking(mc1gt_0),322). orientation(mc1gt_1,65,322). coords(mc1gt_1,243,280,322).  visible(mc1gt_1,322). happensAt(walking(mc1gt_1),322).
orientation(mc1gt_0,10,323). coords(mc1gt_0,213,260,323).  visible(mc1gt_0,323). happensAt(walking(mc1gt_0),323). orientation(mc1gt_1,65,323). coords(mc1gt_1,243,280,323).  visible(mc1gt_1,323). happensAt(walking(mc1gt_1),323).
orientation(mc1gt_0,10,324). coords(mc1gt_0,213,263,324).  visible(mc1gt_0,324). happensAt(walking(mc1gt_0),324). orientation(mc1gt_1,65,324). coords(mc1gt_1,243,280,324).  visible(mc1gt_1,324). happensAt(walking(mc1gt_1),324).
orientation(mc1gt_0,10,325). coords(mc1gt_0,213,266,325).  visible(mc1gt_0,325). happensAt(walking(mc1gt_0),325). orientation(mc1gt_1,65,325). coords(mc1gt_1,243,280,325).  visible(mc1gt_1,325). happensAt(walking(mc1gt_1),325).
orientation(mc1gt_0,10,326). coords(mc1gt_0,213,267,326).  visible(mc1gt_0,326). happensAt(walking(mc1gt_0),326). orientation(mc1gt_1,65,326). coords(mc1gt_1,243,280,326).  visible(mc1gt_1,326). happensAt(walking(mc1gt_1),326).
orientation(mc1gt_0,10,327). coords(mc1gt_0,213,268,327).  visible(mc1gt_0,327). happensAt(walking(mc1gt_0),327). orientation(mc1gt_1,65,327). coords(mc1gt_1,243,280,327).  visible(mc1gt_1,327). happensAt(walking(mc1gt_1),327).
orientation(mc1gt_0,16,328). coords(mc1gt_0,214,269,328).  visible(mc1gt_0,328). happensAt(walking(mc1gt_0),328). orientation(mc1gt_1,65,328). coords(mc1gt_1,244,282,328).  visible(mc1gt_1,328). happensAt(walking(mc1gt_1),328).
orientation(mc1gt_0,19,329). coords(mc1gt_0,214,269,329).  visible(mc1gt_0,329). happensAt(walking(mc1gt_0),329). orientation(mc1gt_1,65,329). coords(mc1gt_1,245,282,329).  visible(mc1gt_1,329). happensAt(walking(mc1gt_1),329).
orientation(mc1gt_0,22,330). coords(mc1gt_0,214,270,330).  visible(mc1gt_0,330). happensAt(walking(mc1gt_0),330). orientation(mc1gt_1,85,330). coords(mc1gt_1,245,284,330).  happensAt(disappear(mc1gt_1),330). happensAt(walking(mc1gt_1),330).
#show holdsAt/2.