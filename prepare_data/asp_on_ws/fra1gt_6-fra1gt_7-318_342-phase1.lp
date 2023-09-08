
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
orientation(fra1gt_6,134,331). coords(fra1gt_6,155,75,331).  visible(fra1gt_6,331). happensAt(walking(fra1gt_6),331). orientation(fra1gt_7,142,331). coords(fra1gt_7,158,65,331).  visible(fra1gt_7,331). happensAt(walking(fra1gt_7),331).
orientation(fra1gt_6,134,332). coords(fra1gt_6,156,74,332).  visible(fra1gt_6,332). happensAt(walking(fra1gt_6),332). orientation(fra1gt_7,142,332). coords(fra1gt_7,160,65,332).  visible(fra1gt_7,332). happensAt(walking(fra1gt_7),332).
orientation(fra1gt_6,134,333). coords(fra1gt_6,157,74,333).  visible(fra1gt_6,333). happensAt(walking(fra1gt_6),333). orientation(fra1gt_7,142,333). coords(fra1gt_7,159,67,333).  visible(fra1gt_7,333). happensAt(walking(fra1gt_7),333).
orientation(fra1gt_6,134,334). coords(fra1gt_6,158,76,334).  visible(fra1gt_6,334). happensAt(walking(fra1gt_6),334). orientation(fra1gt_7,142,334). coords(fra1gt_7,159,68,334).  visible(fra1gt_7,334). happensAt(walking(fra1gt_7),334).
orientation(fra1gt_6,134,335). coords(fra1gt_6,157,76,335).  visible(fra1gt_6,335). happensAt(walking(fra1gt_6),335). orientation(fra1gt_7,142,335). coords(fra1gt_7,161,68,335).  visible(fra1gt_7,335). happensAt(walking(fra1gt_7),335).
orientation(fra1gt_6,134,336). coords(fra1gt_6,157,76,336).  visible(fra1gt_6,336). happensAt(walking(fra1gt_6),336). orientation(fra1gt_7,142,336). coords(fra1gt_7,163,69,336).  visible(fra1gt_7,336). happensAt(walking(fra1gt_7),336).
orientation(fra1gt_6,134,337). coords(fra1gt_6,157,76,337).  visible(fra1gt_6,337). happensAt(walking(fra1gt_6),337). orientation(fra1gt_7,142,337). coords(fra1gt_7,165,69,337).  visible(fra1gt_7,337). happensAt(walking(fra1gt_7),337).
orientation(fra1gt_6,134,338). coords(fra1gt_6,157,77,338).  visible(fra1gt_6,338). happensAt(walking(fra1gt_6),338). orientation(fra1gt_7,142,338). coords(fra1gt_7,165,69,338).  visible(fra1gt_7,338). happensAt(walking(fra1gt_7),338).
orientation(fra1gt_6,134,339). coords(fra1gt_6,158,77,339).  visible(fra1gt_6,339). happensAt(walking(fra1gt_6),339). orientation(fra1gt_7,142,339). coords(fra1gt_7,165,69,339).  visible(fra1gt_7,339). happensAt(walking(fra1gt_7),339).
orientation(fra1gt_6,134,340). coords(fra1gt_6,158,77,340).  visible(fra1gt_6,340). happensAt(walking(fra1gt_6),340). orientation(fra1gt_7,142,340). coords(fra1gt_7,167,70,340).  visible(fra1gt_7,340). happensAt(walking(fra1gt_7),340).
orientation(fra1gt_6,134,341). coords(fra1gt_6,158,77,341).  visible(fra1gt_6,341). happensAt(walking(fra1gt_6),341). orientation(fra1gt_7,142,341). coords(fra1gt_7,168,70,341).  visible(fra1gt_7,341). happensAt(walking(fra1gt_7),341).
orientation(fra1gt_6,134,342). coords(fra1gt_6,160,77,342).  visible(fra1gt_6,342). happensAt(walking(fra1gt_6),342). orientation(fra1gt_7,142,342). coords(fra1gt_7,167,70,342).  visible(fra1gt_7,342). happensAt(walking(fra1gt_7),342).
#show holdsAt/2.