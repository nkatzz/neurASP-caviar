
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

	orientation(mc1gt_0,10,309). coords(mc1gt_0,202,238,309).  visible(mc1gt_0,309). happensAt(walking(mc1gt_0),309). orientation(mc1gt_2,178,309). coords(mc1gt_2,214,203,309).  visible(mc1gt_2,309). happensAt(walking(mc1gt_2),309).
orientation(mc1gt_0,10,310). coords(mc1gt_0,202,242,310).  visible(mc1gt_0,310). happensAt(walking(mc1gt_0),310). orientation(mc1gt_2,178,310). coords(mc1gt_2,214,204,310).  visible(mc1gt_2,310). happensAt(walking(mc1gt_2),310).
orientation(mc1gt_0,10,311). coords(mc1gt_0,204,244,311).  visible(mc1gt_0,311). happensAt(walking(mc1gt_0),311). orientation(mc1gt_2,178,311). coords(mc1gt_2,215,206,311).  visible(mc1gt_2,311). happensAt(walking(mc1gt_2),311).
orientation(mc1gt_0,10,312). coords(mc1gt_0,206,247,312).  visible(mc1gt_0,312). happensAt(walking(mc1gt_0),312). orientation(mc1gt_2,178,312). coords(mc1gt_2,217,208,312).  visible(mc1gt_2,312). happensAt(walking(mc1gt_2),312).
orientation(mc1gt_0,10,313). coords(mc1gt_0,208,249,313).  visible(mc1gt_0,313). happensAt(walking(mc1gt_0),313). orientation(mc1gt_2,178,313). coords(mc1gt_2,217,211,313).  visible(mc1gt_2,313). happensAt(walking(mc1gt_2),313).
orientation(mc1gt_0,10,314). coords(mc1gt_0,209,251,314).  visible(mc1gt_0,314). happensAt(walking(mc1gt_0),314). orientation(mc1gt_2,178,314). coords(mc1gt_2,218,214,314).  visible(mc1gt_2,314). happensAt(walking(mc1gt_2),314).
orientation(mc1gt_0,10,315). coords(mc1gt_0,209,252,315).  visible(mc1gt_0,315). happensAt(walking(mc1gt_0),315). orientation(mc1gt_2,178,315). coords(mc1gt_2,219,218,315).  visible(mc1gt_2,315). happensAt(walking(mc1gt_2),315).
orientation(mc1gt_0,10,316). coords(mc1gt_0,210,253,316).  visible(mc1gt_0,316). happensAt(walking(mc1gt_0),316). orientation(mc1gt_2,178,316). coords(mc1gt_2,221,220,316).  visible(mc1gt_2,316). happensAt(walking(mc1gt_2),316).
orientation(mc1gt_0,10,317). coords(mc1gt_0,211,254,317).  visible(mc1gt_0,317). happensAt(walking(mc1gt_0),317). orientation(mc1gt_2,178,317). coords(mc1gt_2,222,221,317).  visible(mc1gt_2,317). happensAt(walking(mc1gt_2),317).
orientation(mc1gt_0,10,318). coords(mc1gt_0,211,255,318).  visible(mc1gt_0,318). happensAt(walking(mc1gt_0),318). orientation(mc1gt_2,178,318). coords(mc1gt_2,223,222,318).  visible(mc1gt_2,318). happensAt(walking(mc1gt_2),318).
orientation(mc1gt_0,10,319). coords(mc1gt_0,212,255,319).  visible(mc1gt_0,319). happensAt(walking(mc1gt_0),319). orientation(mc1gt_2,178,319). coords(mc1gt_2,224,223,319).  visible(mc1gt_2,319). happensAt(walking(mc1gt_2),319).
orientation(mc1gt_0,10,320). coords(mc1gt_0,213,255,320).  visible(mc1gt_0,320). happensAt(walking(mc1gt_0),320). orientation(mc1gt_2,0,320). coords(mc1gt_2,224,223,320).  visible(mc1gt_2,320). happensAt(walking(mc1gt_2),320).
orientation(mc1gt_0,10,321). coords(mc1gt_0,213,257,321).  visible(mc1gt_0,321). happensAt(walking(mc1gt_0),321). orientation(mc1gt_2,1,321). coords(mc1gt_2,225,224,321).  visible(mc1gt_2,321). happensAt(walking(mc1gt_2),321).
orientation(mc1gt_0,10,322). coords(mc1gt_0,213,258,322).  visible(mc1gt_0,322). happensAt(walking(mc1gt_0),322). orientation(mc1gt_2,4,322). coords(mc1gt_2,225,225,322).  visible(mc1gt_2,322). happensAt(walking(mc1gt_2),322).
orientation(mc1gt_0,10,323). coords(mc1gt_0,213,260,323).  visible(mc1gt_0,323). happensAt(walking(mc1gt_0),323). orientation(mc1gt_2,10,323). coords(mc1gt_2,226,226,323).  visible(mc1gt_2,323). happensAt(walking(mc1gt_2),323).
orientation(mc1gt_0,10,324). coords(mc1gt_0,213,263,324).  visible(mc1gt_0,324). happensAt(walking(mc1gt_0),324). orientation(mc1gt_2,7,324). coords(mc1gt_2,226,230,324).  visible(mc1gt_2,324). happensAt(walking(mc1gt_2),324).
orientation(mc1gt_0,10,325). coords(mc1gt_0,213,266,325).  visible(mc1gt_0,325). happensAt(walking(mc1gt_0),325). orientation(mc1gt_2,13,325). coords(mc1gt_2,229,233,325).  visible(mc1gt_2,325). happensAt(walking(mc1gt_2),325).
orientation(mc1gt_0,10,326). coords(mc1gt_0,213,267,326).  visible(mc1gt_0,326). happensAt(walking(mc1gt_0),326). orientation(mc1gt_2,13,326). coords(mc1gt_2,230,235,326).  visible(mc1gt_2,326). happensAt(walking(mc1gt_2),326).
orientation(mc1gt_0,10,327). coords(mc1gt_0,213,268,327).  visible(mc1gt_0,327). happensAt(walking(mc1gt_0),327). orientation(mc1gt_2,13,327). coords(mc1gt_2,233,238,327).  visible(mc1gt_2,327). happensAt(walking(mc1gt_2),327).
orientation(mc1gt_0,16,328). coords(mc1gt_0,214,269,328).  visible(mc1gt_0,328). happensAt(walking(mc1gt_0),328). orientation(mc1gt_2,13,328). coords(mc1gt_2,236,240,328).  visible(mc1gt_2,328). happensAt(walking(mc1gt_2),328).
orientation(mc1gt_0,19,329). coords(mc1gt_0,214,269,329).  visible(mc1gt_0,329). happensAt(walking(mc1gt_0),329). orientation(mc1gt_2,13,329). coords(mc1gt_2,238,241,329).  visible(mc1gt_2,329). happensAt(walking(mc1gt_2),329).
orientation(mc1gt_0,22,330). coords(mc1gt_0,214,270,330).  visible(mc1gt_0,330). happensAt(walking(mc1gt_0),330). orientation(mc1gt_2,13,330). coords(mc1gt_2,240,243,330).  visible(mc1gt_2,330). happensAt(walking(mc1gt_2),330).
orientation(mc1gt_0,22,331). coords(mc1gt_0,214,272,331).  visible(mc1gt_0,331). happensAt(walking(mc1gt_0),331). orientation(mc1gt_2,13,331). coords(mc1gt_2,242,244,331).  visible(mc1gt_2,331). happensAt(walking(mc1gt_2),331).
orientation(mc1gt_0,22,332). coords(mc1gt_0,214,272,332).  visible(mc1gt_0,332). happensAt(walking(mc1gt_0),332). orientation(mc1gt_2,13,332). coords(mc1gt_2,243,245,332).  visible(mc1gt_2,332). happensAt(walking(mc1gt_2),332).
orientation(mc1gt_0,22,333). coords(mc1gt_0,214,272,333).  visible(mc1gt_0,333). happensAt(walking(mc1gt_0),333). orientation(mc1gt_2,13,333). coords(mc1gt_2,244,245,333).  visible(mc1gt_2,333). happensAt(walking(mc1gt_2),333).
#show holdsAt/2.