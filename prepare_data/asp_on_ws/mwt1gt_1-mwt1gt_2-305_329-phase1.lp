
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

	orientation(mwt1gt_1,160,305). coords(mwt1gt_1,103,165,305).  visible(mwt1gt_1,305). happensAt(walking(mwt1gt_1),305). orientation(mwt1gt_2,140,305). coords(mwt1gt_2,155,99,305).  visible(mwt1gt_2,305). happensAt(walking(mwt1gt_2),305).
orientation(mwt1gt_1,160,306). coords(mwt1gt_1,104,165,306).  visible(mwt1gt_1,306). happensAt(walking(mwt1gt_1),306). orientation(mwt1gt_2,140,306). coords(mwt1gt_2,153,102,306).  visible(mwt1gt_2,306). happensAt(walking(mwt1gt_2),306).
orientation(mwt1gt_1,160,307). coords(mwt1gt_1,106,164,307).  visible(mwt1gt_1,307). happensAt(walking(mwt1gt_1),307). orientation(mwt1gt_2,140,307). coords(mwt1gt_2,153,104,307).  visible(mwt1gt_2,307). happensAt(walking(mwt1gt_2),307).
orientation(mwt1gt_1,160,308). coords(mwt1gt_1,107,164,308).  visible(mwt1gt_1,308). happensAt(walking(mwt1gt_1),308). orientation(mwt1gt_2,140,308). coords(mwt1gt_2,152,104,308).  visible(mwt1gt_2,308). happensAt(walking(mwt1gt_2),308).
orientation(mwt1gt_1,160,309). coords(mwt1gt_1,107,163,309).  visible(mwt1gt_1,309). happensAt(walking(mwt1gt_1),309). orientation(mwt1gt_2,140,309). coords(mwt1gt_2,152,106,309).  visible(mwt1gt_2,309). happensAt(walking(mwt1gt_2),309).
orientation(mwt1gt_1,160,310). coords(mwt1gt_1,107,163,310).  visible(mwt1gt_1,310). happensAt(walking(mwt1gt_1),310). orientation(mwt1gt_2,140,310). coords(mwt1gt_2,152,106,310).  visible(mwt1gt_2,310). happensAt(walking(mwt1gt_2),310).
orientation(mwt1gt_1,160,311). coords(mwt1gt_1,107,162,311).  visible(mwt1gt_1,311). happensAt(walking(mwt1gt_1),311). orientation(mwt1gt_2,140,311). coords(mwt1gt_2,145,108,311).  visible(mwt1gt_2,311). happensAt(walking(mwt1gt_2),311).
orientation(mwt1gt_1,160,312). coords(mwt1gt_1,109,160,312).  visible(mwt1gt_1,312). happensAt(walking(mwt1gt_1),312). orientation(mwt1gt_2,140,312). coords(mwt1gt_2,145,108,312).  visible(mwt1gt_2,312). happensAt(walking(mwt1gt_2),312).
orientation(mwt1gt_1,160,313). coords(mwt1gt_1,109,160,313).  visible(mwt1gt_1,313). happensAt(walking(mwt1gt_1),313). orientation(mwt1gt_2,140,313). coords(mwt1gt_2,143,108,313).  visible(mwt1gt_2,313). happensAt(walking(mwt1gt_2),313).
orientation(mwt1gt_1,160,314). coords(mwt1gt_1,109,160,314).  visible(mwt1gt_1,314). happensAt(walking(mwt1gt_1),314). orientation(mwt1gt_2,140,314). coords(mwt1gt_2,141,109,314).  visible(mwt1gt_2,314). happensAt(walking(mwt1gt_2),314).
orientation(mwt1gt_1,160,315). coords(mwt1gt_1,109,158,315).  visible(mwt1gt_1,315). happensAt(walking(mwt1gt_1),315). orientation(mwt1gt_2,140,315). coords(mwt1gt_2,139,112,315).  visible(mwt1gt_2,315). happensAt(walking(mwt1gt_2),315).
orientation(mwt1gt_1,160,316). coords(mwt1gt_1,109,157,316).  visible(mwt1gt_1,316). happensAt(walking(mwt1gt_1),316). orientation(mwt1gt_2,140,316). coords(mwt1gt_2,138,115,316).  visible(mwt1gt_2,316). happensAt(walking(mwt1gt_2),316).
orientation(mwt1gt_1,160,317). coords(mwt1gt_1,111,155,317).  visible(mwt1gt_1,317). happensAt(walking(mwt1gt_1),317). orientation(mwt1gt_2,140,317). coords(mwt1gt_2,137,116,317).  visible(mwt1gt_2,317). happensAt(walking(mwt1gt_2),317).
orientation(mwt1gt_1,160,318). coords(mwt1gt_1,111,155,318).  visible(mwt1gt_1,318). happensAt(walking(mwt1gt_1),318). orientation(mwt1gt_2,140,318). coords(mwt1gt_2,137,119,318).  visible(mwt1gt_2,318). happensAt(walking(mwt1gt_2),318).
orientation(mwt1gt_1,160,319). coords(mwt1gt_1,114,153,319).  visible(mwt1gt_1,319). happensAt(walking(mwt1gt_1),319). orientation(mwt1gt_2,140,319). coords(mwt1gt_2,137,119,319).  visible(mwt1gt_2,319). happensAt(walking(mwt1gt_2),319).
orientation(mwt1gt_1,160,320). coords(mwt1gt_1,115,153,320).  visible(mwt1gt_1,320). happensAt(walking(mwt1gt_1),320). orientation(mwt1gt_2,140,320). coords(mwt1gt_2,136,120,320).  visible(mwt1gt_2,320). happensAt(walking(mwt1gt_2),320).
orientation(mwt1gt_1,160,321). coords(mwt1gt_1,115,153,321).  visible(mwt1gt_1,321). happensAt(walking(mwt1gt_1),321). orientation(mwt1gt_2,140,321). coords(mwt1gt_2,136,120,321).  visible(mwt1gt_2,321). happensAt(walking(mwt1gt_2),321).
orientation(mwt1gt_1,160,322). coords(mwt1gt_1,115,153,322).  visible(mwt1gt_1,322). happensAt(walking(mwt1gt_1),322). orientation(mwt1gt_2,140,322). coords(mwt1gt_2,133,121,322).  visible(mwt1gt_2,322). happensAt(walking(mwt1gt_2),322).
orientation(mwt1gt_1,160,323). coords(mwt1gt_1,115,153,323).  visible(mwt1gt_1,323). happensAt(walking(mwt1gt_1),323). orientation(mwt1gt_2,140,323). coords(mwt1gt_2,133,122,323).  visible(mwt1gt_2,323). happensAt(walking(mwt1gt_2),323).
orientation(mwt1gt_1,160,324). coords(mwt1gt_1,117,152,324).  visible(mwt1gt_1,324). happensAt(walking(mwt1gt_1),324). orientation(mwt1gt_2,140,324). coords(mwt1gt_2,133,123,324).  visible(mwt1gt_2,324). happensAt(walking(mwt1gt_2),324).
orientation(mwt1gt_1,160,325). coords(mwt1gt_1,118,151,325).  visible(mwt1gt_1,325). happensAt(walking(mwt1gt_1),325). orientation(mwt1gt_2,140,325). coords(mwt1gt_2,133,121,325).  visible(mwt1gt_2,325). happensAt(walking(mwt1gt_2),325).
orientation(mwt1gt_1,160,326). coords(mwt1gt_1,118,151,326).  visible(mwt1gt_1,326). happensAt(walking(mwt1gt_1),326). orientation(mwt1gt_2,140,326). coords(mwt1gt_2,130,123,326).  visible(mwt1gt_2,326). happensAt(walking(mwt1gt_2),326).
orientation(mwt1gt_1,160,327). coords(mwt1gt_1,118,151,327).  visible(mwt1gt_1,327). happensAt(walking(mwt1gt_1),327). orientation(mwt1gt_2,140,327). coords(mwt1gt_2,130,125,327).  visible(mwt1gt_2,327). happensAt(walking(mwt1gt_2),327).
orientation(mwt1gt_1,160,328). coords(mwt1gt_1,118,151,328).  visible(mwt1gt_1,328). happensAt(walking(mwt1gt_1),328). orientation(mwt1gt_2,140,328). coords(mwt1gt_2,130,126,328).  visible(mwt1gt_2,328). happensAt(walking(mwt1gt_2),328).
orientation(mwt1gt_1,160,329). coords(mwt1gt_1,118,151,329).  visible(mwt1gt_1,329). happensAt(walking(mwt1gt_1),329). orientation(mwt1gt_2,140,329). coords(mwt1gt_2,129,126,329).  visible(mwt1gt_2,329). happensAt(walking(mwt1gt_2),329).
#show holdsAt/2.