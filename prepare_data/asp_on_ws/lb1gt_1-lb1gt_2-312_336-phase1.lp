
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

	orientation(lb1gt_1,151,312). coords(lb1gt_1,162,169,312).  visible(lb1gt_1,312). happensAt(walking(lb1gt_1),312). orientation(lb1gt_2,146,312). coords(lb1gt_2,193,151,312).  visible(lb1gt_2,312). happensAt(active(lb1gt_2),312).
orientation(lb1gt_1,151,313). coords(lb1gt_1,162,170,313).  visible(lb1gt_1,313). happensAt(walking(lb1gt_1),313). orientation(lb1gt_2,146,313). coords(lb1gt_2,193,150,313).  visible(lb1gt_2,313). happensAt(active(lb1gt_2),313).
orientation(lb1gt_1,151,314). coords(lb1gt_1,162,170,314).  visible(lb1gt_1,314). happensAt(walking(lb1gt_1),314). orientation(lb1gt_2,146,314). coords(lb1gt_2,192,150,314).  visible(lb1gt_2,314). happensAt(active(lb1gt_2),314).
orientation(lb1gt_1,151,315). coords(lb1gt_1,162,170,315).  visible(lb1gt_1,315). happensAt(walking(lb1gt_1),315). orientation(lb1gt_2,146,315). coords(lb1gt_2,191,150,315).  visible(lb1gt_2,315). happensAt(active(lb1gt_2),315).
orientation(lb1gt_1,151,316). coords(lb1gt_1,163,170,316).  visible(lb1gt_1,316). happensAt(walking(lb1gt_1),316). orientation(lb1gt_2,146,316). coords(lb1gt_2,191,150,316).  visible(lb1gt_2,316). happensAt(active(lb1gt_2),316).
orientation(lb1gt_1,151,317). coords(lb1gt_1,163,170,317).  visible(lb1gt_1,317). happensAt(walking(lb1gt_1),317). orientation(lb1gt_2,146,317). coords(lb1gt_2,191,150,317).  visible(lb1gt_2,317). happensAt(active(lb1gt_2),317).
orientation(lb1gt_1,151,318). coords(lb1gt_1,164,170,318).  visible(lb1gt_1,318). happensAt(walking(lb1gt_1),318). orientation(lb1gt_2,146,318). coords(lb1gt_2,190,150,318).  visible(lb1gt_2,318). happensAt(active(lb1gt_2),318).
orientation(lb1gt_1,151,319). coords(lb1gt_1,164,170,319).  visible(lb1gt_1,319). happensAt(walking(lb1gt_1),319). orientation(lb1gt_2,146,319). coords(lb1gt_2,190,150,319).  visible(lb1gt_2,319). happensAt(active(lb1gt_2),319).
orientation(lb1gt_1,151,320). coords(lb1gt_1,165,170,320).  visible(lb1gt_1,320). happensAt(walking(lb1gt_1),320). orientation(lb1gt_2,146,320). coords(lb1gt_2,190,150,320).  visible(lb1gt_2,320). happensAt(active(lb1gt_2),320).
orientation(lb1gt_1,151,321). coords(lb1gt_1,165,170,321).  visible(lb1gt_1,321). happensAt(walking(lb1gt_1),321). orientation(lb1gt_2,146,321). coords(lb1gt_2,190,150,321).  visible(lb1gt_2,321). happensAt(active(lb1gt_2),321).
orientation(lb1gt_1,151,322). coords(lb1gt_1,165,170,322).  visible(lb1gt_1,322). happensAt(walking(lb1gt_1),322). orientation(lb1gt_2,146,322). coords(lb1gt_2,190,150,322).  visible(lb1gt_2,322). happensAt(active(lb1gt_2),322).
orientation(lb1gt_1,151,323). coords(lb1gt_1,165,170,323).  visible(lb1gt_1,323). happensAt(walking(lb1gt_1),323). orientation(lb1gt_2,146,323). coords(lb1gt_2,190,150,323).  visible(lb1gt_2,323). happensAt(active(lb1gt_2),323).
orientation(lb1gt_1,151,324). coords(lb1gt_1,165,170,324).  visible(lb1gt_1,324). happensAt(active(lb1gt_1),324). orientation(lb1gt_2,146,324). coords(lb1gt_2,190,150,324).  visible(lb1gt_2,324). happensAt(active(lb1gt_2),324).
orientation(lb1gt_1,151,325). coords(lb1gt_1,165,170,325).  visible(lb1gt_1,325). happensAt(active(lb1gt_1),325). orientation(lb1gt_2,146,325). coords(lb1gt_2,190,150,325).  visible(lb1gt_2,325). happensAt(active(lb1gt_2),325).
orientation(lb1gt_1,151,326). coords(lb1gt_1,165,170,326).  visible(lb1gt_1,326). happensAt(active(lb1gt_1),326). orientation(lb1gt_2,146,326). coords(lb1gt_2,190,150,326).  visible(lb1gt_2,326). happensAt(active(lb1gt_2),326).
orientation(lb1gt_1,151,327). coords(lb1gt_1,165,170,327).  visible(lb1gt_1,327). happensAt(active(lb1gt_1),327). orientation(lb1gt_2,146,327). coords(lb1gt_2,190,150,327).  visible(lb1gt_2,327). happensAt(active(lb1gt_2),327).
orientation(lb1gt_1,151,328). coords(lb1gt_1,165,170,328).  visible(lb1gt_1,328). happensAt(active(lb1gt_1),328). orientation(lb1gt_2,146,328). coords(lb1gt_2,190,150,328).  visible(lb1gt_2,328). happensAt(active(lb1gt_2),328).
orientation(lb1gt_1,151,329). coords(lb1gt_1,165,170,329).  visible(lb1gt_1,329). happensAt(active(lb1gt_1),329). orientation(lb1gt_2,146,329). coords(lb1gt_2,190,150,329).  visible(lb1gt_2,329). happensAt(active(lb1gt_2),329).
orientation(lb1gt_1,151,330). coords(lb1gt_1,165,170,330).  visible(lb1gt_1,330). happensAt(active(lb1gt_1),330). orientation(lb1gt_2,146,330). coords(lb1gt_2,190,150,330).  visible(lb1gt_2,330). happensAt(active(lb1gt_2),330).
orientation(lb1gt_1,151,331). coords(lb1gt_1,165,170,331).  visible(lb1gt_1,331). happensAt(active(lb1gt_1),331). orientation(lb1gt_2,146,331). coords(lb1gt_2,190,150,331).  visible(lb1gt_2,331). happensAt(active(lb1gt_2),331).
orientation(lb1gt_1,151,332). coords(lb1gt_1,165,169,332).  visible(lb1gt_1,332). happensAt(active(lb1gt_1),332). orientation(lb1gt_2,146,332). coords(lb1gt_2,190,150,332).  visible(lb1gt_2,332). happensAt(active(lb1gt_2),332).
orientation(lb1gt_1,151,333). coords(lb1gt_1,165,169,333).  visible(lb1gt_1,333). happensAt(active(lb1gt_1),333). orientation(lb1gt_2,146,333). coords(lb1gt_2,190,150,333).  visible(lb1gt_2,333). happensAt(active(lb1gt_2),333).
orientation(lb1gt_1,151,334). coords(lb1gt_1,165,169,334).  visible(lb1gt_1,334). happensAt(active(lb1gt_1),334). orientation(lb1gt_2,146,334). coords(lb1gt_2,190,150,334).  visible(lb1gt_2,334). happensAt(active(lb1gt_2),334).
orientation(lb1gt_1,151,335). coords(lb1gt_1,165,169,335).  visible(lb1gt_1,335). happensAt(active(lb1gt_1),335). orientation(lb1gt_2,146,335). coords(lb1gt_2,190,150,335).  visible(lb1gt_2,335). happensAt(active(lb1gt_2),335).
orientation(lb1gt_1,151,336). coords(lb1gt_1,165,169,336).  visible(lb1gt_1,336). happensAt(active(lb1gt_1),336). orientation(lb1gt_2,146,336). coords(lb1gt_2,190,150,336).  visible(lb1gt_2,336). happensAt(active(lb1gt_2),336).
#show holdsAt/2.