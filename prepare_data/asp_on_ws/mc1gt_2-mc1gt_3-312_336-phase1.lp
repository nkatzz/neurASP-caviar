
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

	orientation(mc1gt_2,178,312). coords(mc1gt_2,217,208,312).  visible(mc1gt_2,312). happensAt(walking(mc1gt_2),312). orientation(mc1gt_3,178,312). coords(mc1gt_3,300,204,312).  visible(mc1gt_3,312). happensAt(walking(mc1gt_3),312).
orientation(mc1gt_2,178,313). coords(mc1gt_2,217,211,313).  visible(mc1gt_2,313). happensAt(walking(mc1gt_2),313). orientation(mc1gt_3,4,313). coords(mc1gt_3,301,206,313).  visible(mc1gt_3,313). happensAt(walking(mc1gt_3),313).
orientation(mc1gt_2,178,314). coords(mc1gt_2,218,214,314).  visible(mc1gt_2,314). happensAt(walking(mc1gt_2),314). orientation(mc1gt_3,25,314). coords(mc1gt_3,301,208,314).  visible(mc1gt_3,314). happensAt(walking(mc1gt_3),314).
orientation(mc1gt_2,178,315). coords(mc1gt_2,219,218,315).  visible(mc1gt_2,315). happensAt(walking(mc1gt_2),315). orientation(mc1gt_3,27,315). coords(mc1gt_3,302,210,315).  visible(mc1gt_3,315). happensAt(walking(mc1gt_3),315).
orientation(mc1gt_2,178,316). coords(mc1gt_2,221,220,316).  visible(mc1gt_2,316). happensAt(walking(mc1gt_2),316). orientation(mc1gt_3,6,316). coords(mc1gt_3,305,215,316).  visible(mc1gt_3,316). happensAt(walking(mc1gt_3),316).
orientation(mc1gt_2,178,317). coords(mc1gt_2,222,221,317).  visible(mc1gt_2,317). happensAt(walking(mc1gt_2),317). orientation(mc1gt_3,3,317). coords(mc1gt_3,306,219,317).  visible(mc1gt_3,317). happensAt(walking(mc1gt_3),317).
orientation(mc1gt_2,178,318). coords(mc1gt_2,223,222,318).  visible(mc1gt_2,318). happensAt(walking(mc1gt_2),318). orientation(mc1gt_3,2,318). coords(mc1gt_3,307,223,318).  visible(mc1gt_3,318). happensAt(walking(mc1gt_3),318).
orientation(mc1gt_2,178,319). coords(mc1gt_2,224,223,319).  visible(mc1gt_2,319). happensAt(walking(mc1gt_2),319). orientation(mc1gt_3,2,319). coords(mc1gt_3,308,225,319).  visible(mc1gt_3,319). happensAt(walking(mc1gt_3),319).
orientation(mc1gt_2,0,320). coords(mc1gt_2,224,223,320).  visible(mc1gt_2,320). happensAt(walking(mc1gt_2),320). orientation(mc1gt_3,3,320). coords(mc1gt_3,309,228,320).  visible(mc1gt_3,320). happensAt(walking(mc1gt_3),320).
orientation(mc1gt_2,1,321). coords(mc1gt_2,225,224,321).  visible(mc1gt_2,321). happensAt(walking(mc1gt_2),321). orientation(mc1gt_3,14,321). coords(mc1gt_3,310,231,321).  visible(mc1gt_3,321). happensAt(walking(mc1gt_3),321).
orientation(mc1gt_2,4,322). coords(mc1gt_2,225,225,322).  visible(mc1gt_2,322). happensAt(walking(mc1gt_2),322). orientation(mc1gt_3,26,322). coords(mc1gt_3,310,234,322).  visible(mc1gt_3,322). happensAt(walking(mc1gt_3),322).
orientation(mc1gt_2,10,323). coords(mc1gt_2,226,226,323).  visible(mc1gt_2,323). happensAt(walking(mc1gt_2),323). orientation(mc1gt_3,34,323). coords(mc1gt_3,310,235,323).  visible(mc1gt_3,323). happensAt(walking(mc1gt_3),323).
orientation(mc1gt_2,7,324). coords(mc1gt_2,226,230,324).  visible(mc1gt_2,324). happensAt(walking(mc1gt_2),324). orientation(mc1gt_3,42,324). coords(mc1gt_3,311,236,324).  visible(mc1gt_3,324). happensAt(walking(mc1gt_3),324).
orientation(mc1gt_2,13,325). coords(mc1gt_2,229,233,325).  visible(mc1gt_2,325). happensAt(walking(mc1gt_2),325). orientation(mc1gt_3,61,325). coords(mc1gt_3,311,236,325).  visible(mc1gt_3,325). happensAt(walking(mc1gt_3),325).
orientation(mc1gt_2,13,326). coords(mc1gt_2,230,235,326).  visible(mc1gt_2,326). happensAt(walking(mc1gt_2),326). orientation(mc1gt_3,61,326). coords(mc1gt_3,312,239,326).  visible(mc1gt_3,326). happensAt(walking(mc1gt_3),326).
orientation(mc1gt_2,13,327). coords(mc1gt_2,233,238,327).  visible(mc1gt_2,327). happensAt(walking(mc1gt_2),327). orientation(mc1gt_3,60,327). coords(mc1gt_3,312,241,327).  visible(mc1gt_3,327). happensAt(walking(mc1gt_3),327).
orientation(mc1gt_2,13,328). coords(mc1gt_2,236,240,328).  visible(mc1gt_2,328). happensAt(walking(mc1gt_2),328). orientation(mc1gt_3,53,328). coords(mc1gt_3,312,244,328).  visible(mc1gt_3,328). happensAt(walking(mc1gt_3),328).
orientation(mc1gt_2,13,329). coords(mc1gt_2,238,241,329).  visible(mc1gt_2,329). happensAt(walking(mc1gt_2),329). orientation(mc1gt_3,56,329). coords(mc1gt_3,312,248,329).  visible(mc1gt_3,329). happensAt(walking(mc1gt_3),329).
orientation(mc1gt_2,13,330). coords(mc1gt_2,240,243,330).  visible(mc1gt_2,330). happensAt(walking(mc1gt_2),330). orientation(mc1gt_3,62,330). coords(mc1gt_3,312,252,330).  visible(mc1gt_3,330). happensAt(walking(mc1gt_3),330).
orientation(mc1gt_2,13,331). coords(mc1gt_2,242,244,331).  visible(mc1gt_2,331). happensAt(walking(mc1gt_2),331). orientation(mc1gt_3,50,331). coords(mc1gt_3,313,256,331).  visible(mc1gt_3,331). happensAt(walking(mc1gt_3),331).
orientation(mc1gt_2,13,332). coords(mc1gt_2,243,245,332).  visible(mc1gt_2,332). happensAt(walking(mc1gt_2),332). orientation(mc1gt_3,51,332). coords(mc1gt_3,313,261,332).  visible(mc1gt_3,332). happensAt(walking(mc1gt_3),332).
orientation(mc1gt_2,13,333). coords(mc1gt_2,244,245,333).  visible(mc1gt_2,333). happensAt(walking(mc1gt_2),333). orientation(mc1gt_3,50,333). coords(mc1gt_3,313,264,333).  visible(mc1gt_3,333). happensAt(walking(mc1gt_3),333).
orientation(mc1gt_2,13,334). coords(mc1gt_2,244,245,334).  visible(mc1gt_2,334). happensAt(walking(mc1gt_2),334). orientation(mc1gt_3,53,334). coords(mc1gt_3,313,265,334).  visible(mc1gt_3,334). happensAt(walking(mc1gt_3),334).
orientation(mc1gt_2,13,335). coords(mc1gt_2,245,246,335).  visible(mc1gt_2,335). happensAt(walking(mc1gt_2),335). orientation(mc1gt_3,58,335). coords(mc1gt_3,313,266,335).  visible(mc1gt_3,335). happensAt(walking(mc1gt_3),335).
orientation(mc1gt_2,13,336). coords(mc1gt_2,245,247,336).  visible(mc1gt_2,336). happensAt(walking(mc1gt_2),336). orientation(mc1gt_3,59,336). coords(mc1gt_3,312,266,336).  visible(mc1gt_3,336). happensAt(walking(mc1gt_3),336).
#show holdsAt/2.