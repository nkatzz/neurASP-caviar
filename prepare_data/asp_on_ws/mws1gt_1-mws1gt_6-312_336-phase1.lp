
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

	orientation(mws1gt_1,155,312). coords(mws1gt_1,80,102,312).  visible(mws1gt_1,312). happensAt(walking(mws1gt_1),312). orientation(mws1gt_6,146,312). coords(mws1gt_6,35,166,312).  visible(mws1gt_6,312). happensAt(walking(mws1gt_6),312).
orientation(mws1gt_1,155,313). coords(mws1gt_1,80,102,313).  visible(mws1gt_1,313). happensAt(walking(mws1gt_1),313). orientation(mws1gt_6,146,313). coords(mws1gt_6,35,166,313).  visible(mws1gt_6,313). happensAt(walking(mws1gt_6),313).
orientation(mws1gt_1,155,314). coords(mws1gt_1,80,102,314).  visible(mws1gt_1,314). happensAt(walking(mws1gt_1),314). orientation(mws1gt_6,146,314). coords(mws1gt_6,35,166,314).  visible(mws1gt_6,314). happensAt(walking(mws1gt_6),314).
orientation(mws1gt_1,155,315). coords(mws1gt_1,78,102,315).  visible(mws1gt_1,315). happensAt(walking(mws1gt_1),315). orientation(mws1gt_6,146,315). coords(mws1gt_6,35,167,315).  visible(mws1gt_6,315). happensAt(walking(mws1gt_6),315).
orientation(mws1gt_1,155,316). coords(mws1gt_1,78,101,316).  visible(mws1gt_1,316). happensAt(walking(mws1gt_1),316). orientation(mws1gt_6,146,316). coords(mws1gt_6,35,167,316).  visible(mws1gt_6,316). happensAt(walking(mws1gt_6),316).
orientation(mws1gt_1,155,317). coords(mws1gt_1,76,101,317).  visible(mws1gt_1,317). happensAt(walking(mws1gt_1),317). orientation(mws1gt_6,146,317). coords(mws1gt_6,35,167,317).  visible(mws1gt_6,317). happensAt(walking(mws1gt_6),317).
orientation(mws1gt_1,155,318). coords(mws1gt_1,76,100,318).  visible(mws1gt_1,318). happensAt(walking(mws1gt_1),318). orientation(mws1gt_6,146,318). coords(mws1gt_6,35,168,318).  visible(mws1gt_6,318). happensAt(walking(mws1gt_6),318).
orientation(mws1gt_1,155,319). coords(mws1gt_1,76,100,319).  visible(mws1gt_1,319). happensAt(walking(mws1gt_1),319). orientation(mws1gt_6,146,319). coords(mws1gt_6,35,169,319).  visible(mws1gt_6,319). happensAt(walking(mws1gt_6),319).
orientation(mws1gt_1,155,320). coords(mws1gt_1,76,100,320).  visible(mws1gt_1,320). happensAt(walking(mws1gt_1),320). orientation(mws1gt_6,146,320). coords(mws1gt_6,35,170,320).  visible(mws1gt_6,320). happensAt(walking(mws1gt_6),320).
orientation(mws1gt_1,155,321). coords(mws1gt_1,76,100,321).  visible(mws1gt_1,321). happensAt(walking(mws1gt_1),321). orientation(mws1gt_6,146,321). coords(mws1gt_6,34,172,321).  visible(mws1gt_6,321). happensAt(walking(mws1gt_6),321).
orientation(mws1gt_1,155,322). coords(mws1gt_1,76,100,322).  visible(mws1gt_1,322). happensAt(walking(mws1gt_1),322). orientation(mws1gt_6,146,322). coords(mws1gt_6,33,174,322).  visible(mws1gt_6,322). happensAt(walking(mws1gt_6),322).
orientation(mws1gt_1,155,323). coords(mws1gt_1,76,100,323).  visible(mws1gt_1,323). happensAt(walking(mws1gt_1),323). orientation(mws1gt_6,146,323). coords(mws1gt_6,33,176,323).  visible(mws1gt_6,323). happensAt(walking(mws1gt_6),323).
orientation(mws1gt_1,155,324). coords(mws1gt_1,75,100,324).  visible(mws1gt_1,324). happensAt(walking(mws1gt_1),324). orientation(mws1gt_6,146,324). coords(mws1gt_6,33,176,324).  visible(mws1gt_6,324). happensAt(walking(mws1gt_6),324).
orientation(mws1gt_1,155,325). coords(mws1gt_1,75,100,325).  visible(mws1gt_1,325). happensAt(walking(mws1gt_1),325). orientation(mws1gt_6,146,325). coords(mws1gt_6,33,176,325).  visible(mws1gt_6,325). happensAt(walking(mws1gt_6),325).
orientation(mws1gt_1,155,326). coords(mws1gt_1,75,100,326).  visible(mws1gt_1,326). happensAt(walking(mws1gt_1),326). orientation(mws1gt_6,146,326). coords(mws1gt_6,33,177,326).  visible(mws1gt_6,326). happensAt(walking(mws1gt_6),326).
orientation(mws1gt_1,155,327). coords(mws1gt_1,75,100,327).  visible(mws1gt_1,327). happensAt(walking(mws1gt_1),327). orientation(mws1gt_6,146,327). coords(mws1gt_6,33,178,327).  visible(mws1gt_6,327). happensAt(walking(mws1gt_6),327).
orientation(mws1gt_1,155,328). coords(mws1gt_1,74,100,328).  visible(mws1gt_1,328). happensAt(walking(mws1gt_1),328). orientation(mws1gt_6,146,328). coords(mws1gt_6,33,178,328).  visible(mws1gt_6,328). happensAt(walking(mws1gt_6),328).
orientation(mws1gt_1,155,329). coords(mws1gt_1,74,99,329).  visible(mws1gt_1,329). happensAt(walking(mws1gt_1),329). orientation(mws1gt_6,146,329). coords(mws1gt_6,33,178,329).  visible(mws1gt_6,329). happensAt(walking(mws1gt_6),329).
orientation(mws1gt_1,155,330). coords(mws1gt_1,73,99,330).  visible(mws1gt_1,330). happensAt(walking(mws1gt_1),330). orientation(mws1gt_6,146,330). coords(mws1gt_6,33,178,330).  visible(mws1gt_6,330). happensAt(walking(mws1gt_6),330).
orientation(mws1gt_1,155,331). coords(mws1gt_1,72,99,331).  visible(mws1gt_1,331). happensAt(walking(mws1gt_1),331). orientation(mws1gt_6,146,331). coords(mws1gt_6,33,179,331).  visible(mws1gt_6,331). happensAt(walking(mws1gt_6),331).
orientation(mws1gt_1,155,332). coords(mws1gt_1,71,99,332).  visible(mws1gt_1,332). happensAt(walking(mws1gt_1),332). orientation(mws1gt_6,146,332). coords(mws1gt_6,33,179,332).  visible(mws1gt_6,332). happensAt(walking(mws1gt_6),332).
orientation(mws1gt_1,155,333). coords(mws1gt_1,71,99,333).  visible(mws1gt_1,333). happensAt(walking(mws1gt_1),333). orientation(mws1gt_6,146,333). coords(mws1gt_6,33,179,333).  visible(mws1gt_6,333). happensAt(walking(mws1gt_6),333).
orientation(mws1gt_1,155,334). coords(mws1gt_1,71,99,334).  visible(mws1gt_1,334). happensAt(walking(mws1gt_1),334). orientation(mws1gt_6,146,334). coords(mws1gt_6,33,180,334).  visible(mws1gt_6,334). happensAt(walking(mws1gt_6),334).
orientation(mws1gt_1,155,335). coords(mws1gt_1,71,99,335).  visible(mws1gt_1,335). happensAt(walking(mws1gt_1),335). orientation(mws1gt_6,146,335). coords(mws1gt_6,32,182,335).  visible(mws1gt_6,335). happensAt(walking(mws1gt_6),335).
orientation(mws1gt_1,155,336). coords(mws1gt_1,71,98,336).  visible(mws1gt_1,336). happensAt(walking(mws1gt_1),336). orientation(mws1gt_6,146,336). coords(mws1gt_6,32,183,336).  visible(mws1gt_6,336). happensAt(walking(mws1gt_6),336).
#show holdsAt/2.