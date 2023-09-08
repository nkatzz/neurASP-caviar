
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

	orientation(wk1gt_1,142,308). coords(wk1gt_1,179,179,308).  visible(wk1gt_1,308). happensAt(walking(wk1gt_1),308). orientation(wk1gt_4,167,308). coords(wk1gt_4,52,173,308).  visible(wk1gt_4,308). happensAt(active(wk1gt_4),308).
orientation(wk1gt_1,151,309). coords(wk1gt_1,178,175,309).  visible(wk1gt_1,309). happensAt(walking(wk1gt_1),309). orientation(wk1gt_4,168,309). coords(wk1gt_4,52,174,309).  visible(wk1gt_4,309). happensAt(active(wk1gt_4),309).
orientation(wk1gt_1,151,310). coords(wk1gt_1,177,173,310).  visible(wk1gt_1,310). happensAt(walking(wk1gt_1),310). orientation(wk1gt_4,164,310). coords(wk1gt_4,52,174,310).  visible(wk1gt_4,310). happensAt(active(wk1gt_4),310).
orientation(wk1gt_1,151,311). coords(wk1gt_1,177,170,311).  visible(wk1gt_1,311). happensAt(walking(wk1gt_1),311). orientation(wk1gt_4,162,311). coords(wk1gt_4,52,175,311).  visible(wk1gt_4,311). happensAt(active(wk1gt_4),311).
orientation(wk1gt_1,151,312). coords(wk1gt_1,176,170,312).  visible(wk1gt_1,312). happensAt(walking(wk1gt_1),312). orientation(wk1gt_4,161,312). coords(wk1gt_4,52,175,312).  visible(wk1gt_4,312). happensAt(active(wk1gt_4),312).
orientation(wk1gt_1,151,313). coords(wk1gt_1,176,169,313).  visible(wk1gt_1,313). happensAt(walking(wk1gt_1),313). orientation(wk1gt_4,162,313). coords(wk1gt_4,52,175,313).  visible(wk1gt_4,313). happensAt(active(wk1gt_4),313).
orientation(wk1gt_1,151,314). coords(wk1gt_1,176,169,314).  visible(wk1gt_1,314). happensAt(walking(wk1gt_1),314). orientation(wk1gt_4,161,314). coords(wk1gt_4,52,176,314).  visible(wk1gt_4,314). happensAt(active(wk1gt_4),314).
orientation(wk1gt_1,151,315). coords(wk1gt_1,175,168,315).  visible(wk1gt_1,315). happensAt(walking(wk1gt_1),315). orientation(wk1gt_4,159,315). coords(wk1gt_4,52,176,315).  visible(wk1gt_4,315). happensAt(inactive(wk1gt_4),315).
orientation(wk1gt_1,151,316). coords(wk1gt_1,175,167,316).  visible(wk1gt_1,316). happensAt(walking(wk1gt_1),316). orientation(wk1gt_4,157,316). coords(wk1gt_4,52,176,316).  visible(wk1gt_4,316). happensAt(inactive(wk1gt_4),316).
orientation(wk1gt_1,151,317). coords(wk1gt_1,174,166,317).  visible(wk1gt_1,317). happensAt(walking(wk1gt_1),317). orientation(wk1gt_4,160,317). coords(wk1gt_4,52,176,317).  visible(wk1gt_4,317). happensAt(inactive(wk1gt_4),317).
orientation(wk1gt_1,151,318). coords(wk1gt_1,173,165,318).  visible(wk1gt_1,318). happensAt(walking(wk1gt_1),318). orientation(wk1gt_4,160,318). coords(wk1gt_4,52,176,318).  visible(wk1gt_4,318). happensAt(inactive(wk1gt_4),318).
orientation(wk1gt_1,151,319). coords(wk1gt_1,171,163,319).  visible(wk1gt_1,319). happensAt(walking(wk1gt_1),319). orientation(wk1gt_4,160,319). coords(wk1gt_4,52,176,319).  visible(wk1gt_4,319). happensAt(inactive(wk1gt_4),319).
orientation(wk1gt_1,151,320). coords(wk1gt_1,170,161,320).  visible(wk1gt_1,320). happensAt(walking(wk1gt_1),320). orientation(wk1gt_4,160,320). coords(wk1gt_4,52,176,320).  visible(wk1gt_4,320). happensAt(inactive(wk1gt_4),320).
orientation(wk1gt_1,158,321). coords(wk1gt_1,168,159,321).  visible(wk1gt_1,321). happensAt(walking(wk1gt_1),321). orientation(wk1gt_4,161,321). coords(wk1gt_4,52,176,321).  visible(wk1gt_4,321). happensAt(inactive(wk1gt_4),321).
orientation(wk1gt_1,158,322). coords(wk1gt_1,167,159,322).  visible(wk1gt_1,322). happensAt(walking(wk1gt_1),322). orientation(wk1gt_4,162,322). coords(wk1gt_4,52,176,322).  visible(wk1gt_4,322). happensAt(inactive(wk1gt_4),322).
orientation(wk1gt_1,158,323). coords(wk1gt_1,165,155,323).  visible(wk1gt_1,323). happensAt(walking(wk1gt_1),323). orientation(wk1gt_4,161,323). coords(wk1gt_4,52,176,323).  visible(wk1gt_4,323). happensAt(inactive(wk1gt_4),323).
orientation(wk1gt_1,158,324). coords(wk1gt_1,164,155,324).  visible(wk1gt_1,324). happensAt(walking(wk1gt_1),324). orientation(wk1gt_4,161,324). coords(wk1gt_4,52,176,324).  visible(wk1gt_4,324). happensAt(inactive(wk1gt_4),324).
orientation(wk1gt_1,158,325). coords(wk1gt_1,163,154,325).  visible(wk1gt_1,325). happensAt(walking(wk1gt_1),325). orientation(wk1gt_4,158,325). coords(wk1gt_4,52,176,325).  visible(wk1gt_4,325). happensAt(inactive(wk1gt_4),325).
orientation(wk1gt_1,158,326). coords(wk1gt_1,162,154,326).  visible(wk1gt_1,326). happensAt(walking(wk1gt_1),326). orientation(wk1gt_4,166,326). coords(wk1gt_4,52,176,326).  visible(wk1gt_4,326). happensAt(inactive(wk1gt_4),326).
orientation(wk1gt_1,150,327). coords(wk1gt_1,162,153,327).  visible(wk1gt_1,327). happensAt(walking(wk1gt_1),327). orientation(wk1gt_4,162,327). coords(wk1gt_4,52,176,327).  visible(wk1gt_4,327). happensAt(inactive(wk1gt_4),327).
orientation(wk1gt_1,150,328). coords(wk1gt_1,162,153,328).  visible(wk1gt_1,328). happensAt(walking(wk1gt_1),328). orientation(wk1gt_4,162,328). coords(wk1gt_4,52,176,328).  visible(wk1gt_4,328). happensAt(inactive(wk1gt_4),328).
orientation(wk1gt_1,150,329). coords(wk1gt_1,162,152,329).  visible(wk1gt_1,329). happensAt(walking(wk1gt_1),329). orientation(wk1gt_4,158,329). coords(wk1gt_4,52,176,329).  visible(wk1gt_4,329). happensAt(inactive(wk1gt_4),329).
orientation(wk1gt_1,148,330). coords(wk1gt_1,160,151,330).  visible(wk1gt_1,330). happensAt(walking(wk1gt_1),330). orientation(wk1gt_4,160,330). coords(wk1gt_4,52,176,330).  visible(wk1gt_4,330). happensAt(inactive(wk1gt_4),330).
orientation(wk1gt_1,148,331). coords(wk1gt_1,159,150,331).  visible(wk1gt_1,331). happensAt(walking(wk1gt_1),331). orientation(wk1gt_4,157,331). coords(wk1gt_4,52,176,331).  visible(wk1gt_4,331). happensAt(inactive(wk1gt_4),331).
orientation(wk1gt_1,148,332). coords(wk1gt_1,156,148,332).  visible(wk1gt_1,332). happensAt(walking(wk1gt_1),332). orientation(wk1gt_4,160,332). coords(wk1gt_4,52,176,332).  visible(wk1gt_4,332). happensAt(inactive(wk1gt_4),332).
#show holdsAt/2.