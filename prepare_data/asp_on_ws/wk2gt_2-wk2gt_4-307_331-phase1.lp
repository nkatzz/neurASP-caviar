
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

	orientation(wk2gt_2,141,307). coords(wk2gt_2,98,63,307).  visible(wk2gt_2,307). happensAt(inactive(wk2gt_2),307). orientation(wk2gt_4,47,307). coords(wk2gt_4,257,266,307).  visible(wk2gt_4,307). happensAt(walking(wk2gt_4),307).
orientation(wk2gt_2,141,308). coords(wk2gt_2,98,63,308).  visible(wk2gt_2,308). happensAt(inactive(wk2gt_2),308). orientation(wk2gt_4,36,308). coords(wk2gt_4,256,264,308).  visible(wk2gt_4,308). happensAt(walking(wk2gt_4),308).
orientation(wk2gt_2,141,309). coords(wk2gt_2,98,63,309).  visible(wk2gt_2,309). happensAt(inactive(wk2gt_2),309). orientation(wk2gt_4,36,309). coords(wk2gt_4,255,259,309).  visible(wk2gt_4,309). happensAt(walking(wk2gt_4),309).
orientation(wk2gt_2,141,310). coords(wk2gt_2,98,63,310).  visible(wk2gt_2,310). happensAt(inactive(wk2gt_2),310). orientation(wk2gt_4,36,310). coords(wk2gt_4,255,255,310).  visible(wk2gt_4,310). happensAt(walking(wk2gt_4),310).
orientation(wk2gt_2,141,311). coords(wk2gt_2,98,63,311).  visible(wk2gt_2,311). happensAt(inactive(wk2gt_2),311). orientation(wk2gt_4,36,311). coords(wk2gt_4,254,252,311).  visible(wk2gt_4,311). happensAt(walking(wk2gt_4),311).
orientation(wk2gt_2,141,312). coords(wk2gt_2,98,63,312).  visible(wk2gt_2,312). happensAt(inactive(wk2gt_2),312). orientation(wk2gt_4,36,312). coords(wk2gt_4,254,249,312).  visible(wk2gt_4,312). happensAt(walking(wk2gt_4),312).
orientation(wk2gt_2,141,313). coords(wk2gt_2,98,63,313).  visible(wk2gt_2,313). happensAt(inactive(wk2gt_2),313). orientation(wk2gt_4,36,313). coords(wk2gt_4,253,249,313).  visible(wk2gt_4,313). happensAt(walking(wk2gt_4),313).
orientation(wk2gt_2,141,314). coords(wk2gt_2,98,63,314).  visible(wk2gt_2,314). happensAt(inactive(wk2gt_2),314). orientation(wk2gt_4,23,314). coords(wk2gt_4,252,247,314).  visible(wk2gt_4,314). happensAt(walking(wk2gt_4),314).
orientation(wk2gt_2,141,315). coords(wk2gt_2,98,63,315).  visible(wk2gt_2,315). happensAt(inactive(wk2gt_2),315). orientation(wk2gt_4,9,315). coords(wk2gt_4,251,246,315).  visible(wk2gt_4,315). happensAt(walking(wk2gt_4),315).
orientation(wk2gt_2,141,316). coords(wk2gt_2,98,63,316).  visible(wk2gt_2,316). happensAt(inactive(wk2gt_2),316). orientation(wk2gt_4,2,316). coords(wk2gt_4,250,244,316).  visible(wk2gt_4,316). happensAt(walking(wk2gt_4),316).
orientation(wk2gt_2,141,317). coords(wk2gt_2,98,63,317).  visible(wk2gt_2,317). happensAt(inactive(wk2gt_2),317). orientation(wk2gt_4,2,317). coords(wk2gt_4,249,242,317).  visible(wk2gt_4,317). happensAt(walking(wk2gt_4),317).
orientation(wk2gt_2,141,318). coords(wk2gt_2,98,63,318).  visible(wk2gt_2,318). happensAt(inactive(wk2gt_2),318). orientation(wk2gt_4,2,318). coords(wk2gt_4,245,240,318).  visible(wk2gt_4,318). happensAt(walking(wk2gt_4),318).
orientation(wk2gt_2,141,319). coords(wk2gt_2,98,63,319).  visible(wk2gt_2,319). happensAt(inactive(wk2gt_2),319). orientation(wk2gt_4,2,319). coords(wk2gt_4,243,237,319).  visible(wk2gt_4,319). happensAt(walking(wk2gt_4),319).
orientation(wk2gt_2,141,320). coords(wk2gt_2,98,63,320).  visible(wk2gt_2,320). happensAt(inactive(wk2gt_2),320). orientation(wk2gt_4,2,320). coords(wk2gt_4,241,235,320).  visible(wk2gt_4,320). happensAt(walking(wk2gt_4),320).
orientation(wk2gt_2,141,321). coords(wk2gt_2,98,63,321).  visible(wk2gt_2,321). happensAt(inactive(wk2gt_2),321). orientation(wk2gt_4,2,321). coords(wk2gt_4,239,230,321).  visible(wk2gt_4,321). happensAt(walking(wk2gt_4),321).
orientation(wk2gt_2,141,322). coords(wk2gt_2,98,63,322).  visible(wk2gt_2,322). happensAt(inactive(wk2gt_2),322). orientation(wk2gt_4,2,322). coords(wk2gt_4,238,226,322).  visible(wk2gt_4,322). happensAt(walking(wk2gt_4),322).
orientation(wk2gt_2,141,323). coords(wk2gt_2,98,63,323).  visible(wk2gt_2,323). happensAt(inactive(wk2gt_2),323). orientation(wk2gt_4,174,323). coords(wk2gt_4,237,222,323).  visible(wk2gt_4,323). happensAt(walking(wk2gt_4),323).
orientation(wk2gt_2,141,324). coords(wk2gt_2,98,63,324).  visible(wk2gt_2,324). happensAt(inactive(wk2gt_2),324). orientation(wk2gt_4,167,324). coords(wk2gt_4,237,220,324).  visible(wk2gt_4,324). happensAt(walking(wk2gt_4),324).
orientation(wk2gt_2,141,325). coords(wk2gt_2,98,63,325).  visible(wk2gt_2,325). happensAt(inactive(wk2gt_2),325). orientation(wk2gt_4,158,325). coords(wk2gt_4,237,219,325).  visible(wk2gt_4,325). happensAt(walking(wk2gt_4),325).
orientation(wk2gt_2,141,326). coords(wk2gt_2,98,63,326).  visible(wk2gt_2,326). happensAt(inactive(wk2gt_2),326). orientation(wk2gt_4,158,326). coords(wk2gt_4,235,219,326).  visible(wk2gt_4,326). happensAt(walking(wk2gt_4),326).
orientation(wk2gt_2,141,327). coords(wk2gt_2,98,63,327).  visible(wk2gt_2,327). happensAt(inactive(wk2gt_2),327). orientation(wk2gt_4,158,327). coords(wk2gt_4,234,218,327).  visible(wk2gt_4,327). happensAt(walking(wk2gt_4),327).
orientation(wk2gt_2,141,328). coords(wk2gt_2,98,63,328).  visible(wk2gt_2,328). happensAt(inactive(wk2gt_2),328). orientation(wk2gt_4,158,328). coords(wk2gt_4,233,217,328).  visible(wk2gt_4,328). happensAt(walking(wk2gt_4),328).
orientation(wk2gt_2,141,329). coords(wk2gt_2,98,63,329).  visible(wk2gt_2,329). happensAt(inactive(wk2gt_2),329). orientation(wk2gt_4,158,329). coords(wk2gt_4,230,215,329).  visible(wk2gt_4,329). happensAt(walking(wk2gt_4),329).
orientation(wk2gt_2,141,330). coords(wk2gt_2,98,63,330).  visible(wk2gt_2,330). happensAt(inactive(wk2gt_2),330). orientation(wk2gt_4,158,330). coords(wk2gt_4,228,212,330).  visible(wk2gt_4,330). happensAt(walking(wk2gt_4),330).
orientation(wk2gt_2,141,331). coords(wk2gt_2,98,63,331).  visible(wk2gt_2,331). happensAt(inactive(wk2gt_2),331). orientation(wk2gt_4,158,331). coords(wk2gt_4,226,208,331).  visible(wk2gt_4,331). happensAt(walking(wk2gt_4),331).
#show holdsAt/2.